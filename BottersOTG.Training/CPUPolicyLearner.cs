using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BottersOTG.Intelligence.Decisions;
using BottersOTG.Model;
using Utils;

namespace BottersOTG.Training {
	public class CPUPolicyLearner {
		private const double RequiredIncreaseToSplit = 0.001;

		private readonly ContinuousDecisionLearner _continuousLearner = new ContinuousDecisionLearner();
		private readonly CategoricalDecisionLearner _categoricalLearner = new CategoricalDecisionLearner();

		public async Task<Policy> FitPolicy(List<Episode> episodes) {
			Policy policy = new Policy();

			foreach (var heroEpisodeGroup in episodes.AsParallel().GroupBy(x => x.Hero.HeroType)) {
				HeroType heroType = heroEpisodeGroup.Key;
				List<Episode> heroEpisodes = heroEpisodeGroup.ToList();

				policy.Root[heroType] = await FitPolicyToHero(heroType, heroEpisodes, policy);
			}

			return policy;
		}

		private async Task<IDecisionNode> FitPolicyToHero(HeroType heroType, List<Episode> episodes, Policy policy) {
			double grandTotalWeight = episodes.Sum(x => x.Weight);
			DecisionNodeAccuracy decisionNodeAccuracy =
				await OptimalDecisionNode(episodes, grandTotalWeight);
			Console.WriteLine(string.Format(
				"CPU: Policy [{0}] ({1} leaves from {2} episodes) accuracy: {3}",
				heroType == HeroType.None ? "All" : heroType.ToString(),
				CountLeaves(decisionNodeAccuracy.DecisionNode),
				episodes.Count,
				decisionNodeAccuracy.CorrectWeight / grandTotalWeight));

			return decisionNodeAccuracy.DecisionNode;
		}

		private async Task<DecisionNodeAccuracy> OptimalDecisionNode(List<Episode> episodes, double grandTotalWeight) {
			double[] tacticWeights = TacticEntropy.TacticFrequency(episodes);
			double totalWeight = tacticWeights.Sum();
			double leafEntropy = TacticEntropy.Entropy(tacticWeights);

			DecisionNodeAccuracy bestLeaf = OptimalLeaf(episodes, tacticWeights);
			if (bestLeaf.CorrectWeight + double.Epsilon >= totalWeight) {
				// We already know the perfect answer
				return bestLeaf;
			}

			double maximumCorrectWeight = totalWeight;
			double maximumIncrease = maximumCorrectWeight - bestLeaf.CorrectWeight;
			double requiredIncrease = RequiredIncreaseToSplit * grandTotalWeight;
			if (maximumIncrease < requiredIncrease) {
				// Cannot make a big enough difference to the big picture
				return bestLeaf;
			}



			PartitionScore partitionScore =
				_continuousLearner.Optimizers(episodes, tacticWeights)
				.Concat(_categoricalLearner.Optimizers(episodes, tacticWeights))
				.AsParallel()
				.Select(partitionEvaluator => partitionEvaluator())
				.Where(x => x != null && x.LeftEpisodes.Count > 0 && x.RightEpisodes.Count > 0)
				.MinByOrDefault(x => x.Entropy);
			if (partitionScore == null || partitionScore.Entropy >= leafEntropy) {
				if (partitionScore?.Entropy > leafEntropy + double.Epsilon) {
					Console.WriteLine("Entropy should never increase with a split");
				}
				// No split found
				return bestLeaf;
			}

			DecisionNodeAccuracy left = await OptimalDecisionNode(partitionScore.LeftEpisodes, grandTotalWeight);
			DecisionNodeAccuracy right = await OptimalDecisionNode(partitionScore.RightEpisodes, grandTotalWeight);

			DecisionNode splitter = new DecisionNode() {
				Partitioner = partitionScore.Partitioner,
				Left = left.DecisionNode,
				Right = right.DecisionNode,
			};
			double bestSplitCorrectWeight = left.CorrectWeight + right.CorrectWeight;
			double correctWeightIncrease = bestSplitCorrectWeight - bestLeaf.CorrectWeight;

			if (correctWeightIncrease < requiredIncrease) {
				return bestLeaf;
			} else {
				return new DecisionNodeAccuracy {
					DecisionNode = splitter,
					CorrectWeight = bestSplitCorrectWeight,
				};
			}
		}

		private DecisionNodeAccuracy OptimalLeaf(List<Episode> episodes, double[] tacticWeights) {
			TacticWeight bestTacticOverall =
				tacticWeights
				.Select((weight, i) => new TacticWeight { Tactic = (Tactic)i, Weight = weight })
				.MaxByOrDefault(x => x.Weight);
			if (!IsSpell(bestTacticOverall.Tactic)) {
				return new DecisionNodeAccuracy {
					DecisionNode = new DecisionLeaf(bestTacticOverall.Tactic),
					CorrectWeight = bestTacticOverall.Weight,
				};
			}

			Dictionary<HeroType, Tactic> bestTacticPerHero = new Dictionary<HeroType, Tactic>();
			double totalCorrectWeight = 0.0;
			foreach (var group in episodes.GroupBy(ep => ep.Hero.HeroType)) {
				TacticWeight bestTacticForHero =
					TacticEntropy.TacticFrequency(group)
					.Select((weight, i) => new TacticWeight { Tactic = (Tactic)i, Weight = weight })
					.MaxByOrDefault(x => x.Weight);
				bestTacticPerHero[group.Key] = bestTacticForHero.Tactic;
				totalCorrectWeight += bestTacticForHero.Weight;
			}

			foreach (HeroType heroType in EnumUtils.GetEnumValues<HeroType>()) {
				if (bestTacticPerHero.GetOrDefault(heroType) == bestTacticOverall.Tactic) {
					bestTacticPerHero.Remove(heroType);
				}
			}

			return new DecisionNodeAccuracy {
				DecisionNode = new DecisionLeaf(bestTacticOverall.Tactic, bestTacticPerHero),
				CorrectWeight = totalCorrectWeight,
			};
		}

		private bool IsSpell(Tactic tactic) {
			switch (tactic) {
				case Tactic.AttackHero:
				case Tactic.AttackSafely:
				case Tactic.Retreat:
					return false;
				default:
					return true;
			}
		}

		private int CountLeaves(IDecisionNode node) {
			if (node is DecisionLeaf) {
				return 1;
			} else if (node is DecisionNode) {
				DecisionNode decisionNode = (DecisionNode)node;
				return CountLeaves(decisionNode.Left) + CountLeaves(decisionNode.Right);
			} else {
				throw new ArgumentException("Unknown node type: " + node);
			}
		}

		private class DecisionNodeAccuracy {
			public IDecisionNode DecisionNode;
			public double CorrectWeight;
		}

		private class TacticWeight {
			public Tactic Tactic;
			public double Weight;
		}
	}
}
