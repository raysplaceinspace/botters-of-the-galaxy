using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using BottersOTG.Intelligence.Decisions;
using BottersOTG.Model;
using Telogis.RouteCloud.GPUManagement;
using Utils;

namespace BottersOTG.Training {
	class Trainer : IDisposable {
		public const int StatusInterval = 10000; 
		public const int ReplaysPerIteration = 25000;

		private readonly ThreadLocal<ThreadContext> _threadContext = new ThreadLocal<ThreadContext>(() => new ThreadContext());
		private readonly GPUPolicyLearner _policyLearner;

		private readonly Policy[] _starterPolicies = new[] {
			new Policy { Default = new DecisionLeaf(Tactic.Retreat) },
			new Policy { Default = new DecisionLeaf(Tactic.AttackHero) },
		};

		public Trainer() {
			_policyLearner = new GPUPolicyLearner();
		}

		public TrainingResult StrengthenPolicy(Policy initialPolicy, Policy incumbent) {
			Stopwatch initialisingStopwatch = Stopwatch.StartNew();
			Console.WriteLine(string.Format("Generating initial worlds..."));
			PolicyCandidate initial = EvaluatePolicy(initialPolicy, incumbent);
			Rollout[] starters =
				initial.Rollouts
				.Concat(_starterPolicies.SelectMany(starter => EvaluatePolicy(starter, incumbent).Rollouts))
				.ToArray();
			Console.WriteLine(string.Format("{0} initial matchups generated in {1:F1} s", initial.Rollouts.Length, initialisingStopwatch.Elapsed.TotalSeconds));

			// Generate new policy
			PolicyCandidate best = initial;
			while (true) {
				HeroType chosenHero = ChooseHero();
				PolicyCandidate candidate = GeneratePolicyCandidate(best.Policy, incumbent, starters, chosenHero);
				Console.WriteLine(string.Format("Candidate: {0} -> {1}", initial.WinRate, candidate.WinRate));

				// Update best
				if (candidate.WinRate >= best.WinRate) {
					best = candidate;
				}
				if (best.WinRate > 0.5 && best.WinRate > initial.WinRate) {
					break;
				}
			}

			Console.WriteLine(string.Format("Win rate: {0} -> {1}", initial.WinRate, best.WinRate));
			return new TrainingResult {
				Policy = best.Policy,
				WinRate = best.WinRate,
				IsImprovement = best.WinRate > initial.WinRate,
			};
		}

		private HeroType ChooseHero() {
			HeroType[] heroes = EnumUtils.GetEnumValues<HeroType>().Where(h => h != HeroType.None).ToArray();
			return heroes[_threadContext.Value.Random.Next(heroes.Length)];
		}

		private PolicyCandidate EvaluatePolicy(Policy policy, Policy incumbent) {
			Rollout[] rollouts = RolloutPerformer.Rollout(policy, incumbent);
			return new PolicyCandidate {
				Policy = policy,
				Rollouts = rollouts,
				WinRate = rollouts.Average(r => IntermediateWinner(r)),
			};
		}

		private double IntermediateWinner(Rollout rollout) {
			double team0 = IntermediateEvaluator.Evaluate(rollout.FinalWorld, 0).Score;
			double team1 = IntermediateEvaluator.Evaluate(rollout.FinalWorld, 1).Score;
			if (team0 > team1) {
				return 1;
			} else if (team0 < team1) {
				return 0;
			} else {
				return 0.5;
			}
		}

		private List<Episode> EpisodesFromBestRollouts(Rollout[] bestRollouts, HeroType chosenHero) {
			return Enumerable.Range(0, ReplaysPerIteration).AsParallel()
				.Select(_ => {
					Random random = _threadContext.Value.Random;

					while (true) {
						Rollout rollout = bestRollouts[random.Next(bestRollouts.Length)];
						RolloutTick tick = rollout.Ticks[random.Next(rollout.Ticks.Count)];
						World world = tick.World;
						Unit hero = world.Units.FirstOrDefault(u => u.Team == 0 && u.UnitType == UnitType.Hero && u.HeroType == chosenHero);
						if (hero == null) {
							continue;
						}

						return new Episode {
							Hero = hero,
							Tactic = tick.HeroTactics[hero.UnitId],
							Weight = 1.0,
							World = world,
						};
					}
				})
				.ToList();
		}

		private PolicyCandidate GeneratePolicyCandidate(Policy currentPolicy, Policy incumbent, Rollout[] starters, HeroType chosenHero) {
			List<Episode> trainingSet = GenerateTrainingSet(incumbent, currentPolicy, starters, chosenHero);
			Policy newPolicy = FitPolicy(currentPolicy, trainingSet);
			return EvaluatePolicy(newPolicy, incumbent);
		}

		private Policy FitPolicy(Policy currentPolicy, List<Episode> trainingSet) {
			Stopwatch fittingStopwatch = Stopwatch.StartNew();
			Console.WriteLine(string.Format("Fitting policy..."));
			Policy newPolicy;
			{
				Policy fitPolicy = _policyLearner.FitPolicy(trainingSet);
				newPolicy = currentPolicy.Clone();
				foreach (HeroType hero in fitPolicy.Root.Keys) {
					newPolicy.Root[hero] = fitPolicy.Root[hero];
				}
			}

			Console.WriteLine(string.Format("Policy generated in {0:F1} seconds", fittingStopwatch.Elapsed.TotalSeconds));

			Stopwatch verifyingStopwatch = Stopwatch.StartNew();
			Console.WriteLine(string.Format("Verifying policy..."));
			double policyAccuracy = CalculatePolicyAccuracy(trainingSet, newPolicy);
			Console.WriteLine(string.Format("Policy accuracy: {0}", policyAccuracy));
			return newPolicy;
		}

		private List<Episode> GenerateTrainingSet(Policy incumbent, Policy currentPolicy, Rollout[] starters, HeroType chosenHero) {
			List<Episode> trainingSet = new List<Episode>();
			int generated = trainingSet.Count;
			int nextMessage = NextStatusInterval(generated, StatusInterval);

			Stopwatch generatingStopwatch = Stopwatch.StartNew();
			Console.WriteLine(string.Format("Generating up to {0} replays", ReplaysPerIteration));
			Console.Write("> ");
			trainingSet.AddRange(
				Enumerable.Range(0, ReplaysPerIteration)
				.AsParallel()
				.SelectMany(_ => {
					Random random = _threadContext.Value.Random;

					while (true) {
						Rollout rollout = starters[random.Next(starters.Length)];
						World initialWorld = rollout.Ticks[random.Next(rollout.Ticks.Count)].World;
						Unit hero = initialWorld.Units.FirstOrDefault(u => u.Team == 0 && u.UnitType == UnitType.Hero && u.HeroType == chosenHero);
						if (hero == null) {
							continue;
						}

						List<Episode> episodes = Replayer.GenerateRollout(initialWorld, hero.UnitId, currentPolicy, incumbent);

						int nowGenerated = Interlocked.Add(ref generated, episodes.Count);
						int nowMessage = nextMessage;
						int newMessage = NextStatusInterval(nowGenerated, StatusInterval);
						if (nowGenerated >= nowMessage &&
							Interlocked.CompareExchange(ref nextMessage, newMessage, nowMessage) == nowMessage) {
							Console.Write(nowMessage + " ");
						}
						return episodes;
					}
				})
				.WhereNotNull());
			Console.WriteLine();
			Console.WriteLine(string.Format("{0} episodes generated in {1:F1} seconds", trainingSet.Count, generatingStopwatch.Elapsed.TotalSeconds));
			return trainingSet;
		}

		private static int NextStatusInterval(int generated, int statusInterval) {
			return (int)Math.Ceiling((double)(generated + 1) / statusInterval) * statusInterval;
		}

		private double CalculatePolicyAccuracy(List<Episode> trainingSet, Policy policy) {
			double correct =
				trainingSet.AsParallel()
				.Sum(episode => PolicyEvaluator.Evaluate(episode.World, episode.Hero, policy) == episode.Tactic ? episode.Weight : 0);
			double total = trainingSet.Sum(ep => ep.Weight);
			return correct / total;
		}

		public void Dispose() {
			_policyLearner?.Dispose();
			_threadContext?.Dispose();
		}

		private class ThreadContext {
			public Random Random = new Random();
		}

		private class PolicyCandidate {
			public Policy Policy;
			public Rollout[] Rollouts;
			public double WinRate;
		}
	}
}
