using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BottersOTG.CodinGame;
using BottersOTG.Intelligence.Decisions;
using BottersOTG.Model;
using Utils;

namespace BottersOTG.Training {
	class Program {
		public static void Main(string[] args) {
#if DEBUGCUDA
			Telogis.RouteCloud.GPUManagement.KernelManager.SynchroniseAfterEveryKernel = true;
#endif

			string outputPath = args[0];

			Console.WriteLine("Environment.ProcessorCount: " + Environment.ProcessorCount);
			Console.WriteLine("64-bit process? " + Environment.Is64BitProcess);

			Console.WriteLine("World.NumHeroesPerTeam = " + World.NumHeroesPerTeam);
			Console.WriteLine("World.SpawnMinions = " + World.SpawnMinions);
			Console.WriteLine("World.EnableBuySell = " + World.EnableBuySell);
			Console.WriteLine("World.EnableSpells = " + World.EnableSpells);

			using (Trainer trainer = new Trainer()) {
				Policy policy = PolicyProvider.Policy;

				Policy[] incumbents = SubmittedPolicyProvider.Submissions.ToArray();
				double[] incumbentWinRates =
					incumbents
					.Select(oldPolicy => RolloutPerformer.Winner(policy, oldPolicy).Average(x => x.WinRate)).ToArray();
				Console.WriteLine("Initial win rates: " + string.Join(" ", incumbentWinRates));

				int improvements = 0;
				for (int iteration = 0; ; ++iteration) {
					Policy incumbent = policy;
					Console.WriteLine(string.Format("Iteration #{0}", iteration));

					TrainingResult trainingResult = trainer.StrengthenPolicy(policy, incumbent);
					policy = trainingResult.Policy;

					List<Rollout[]> incumbentResults = incumbents.Select(oldPolicy => RolloutPerformer.Winner(policy, oldPolicy)).ToList();
					incumbentWinRates = incumbentResults.Select(submissionResult => submissionResult.Average(x => x.WinRate)).ToArray();
					Console.WriteLine("Against incumbents: " + string.Join(" ", incumbentWinRates));

					policy.HeroMatchups = MatchupOptimizer.Optimize(incumbentResults.SelectMany(x => x));

					if (trainingResult.IsImprovement) {
						++improvements;
						Console.WriteLine(string.Format("Improvement #{0}", improvements));
					}

					if (outputPath != null) {
						File.WriteAllText(outputPath, PolicySerializer.Serialize(policy));
					}

					Console.WriteLine();
				}
			}
		}
	}
}
