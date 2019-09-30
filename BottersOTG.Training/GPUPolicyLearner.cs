using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BottersOTG.Intelligence.Decisions;
using BottersOTG.Model;
using BottersOTG.Training.DecisionLearning;
using BottersOTG.Training.DecisionLearning.Model;
using Telogis.RouteCloud.GPUManagement;
using Utils;

namespace BottersOTG.Training {
	public class GPUPolicyLearner : IDisposable {
		private readonly CudaManager _cudaManager;

		public GPUPolicyLearner() {
			_cudaManager = Provider.CudaManagerPool.GetCudaManagerForThread(Provider.Logger);
		}

		public Policy FitPolicy(List<Episode> episodes) {
			Policy gpuPolicy = new Policy();

			foreach (var heroEpisodeGroup in episodes.AsParallel().GroupBy(x => x.Hero.HeroType)) {
				HeroType heroType = heroEpisodeGroup.Key;
				List<Episode> heroEpisodes = heroEpisodeGroup.ToList();

				if (heroEpisodes.Count > 0) {
					gpuPolicy.Root[heroType] = FitPolicyToHero(heroType, heroEpisodes, gpuPolicy);
				}
			}

#if DEBUGCUDA
			// Task<Policy> cpuPolicyTask = new CPUPolicyLearner().FitPolicy(episodes);
			// cpuPolicyTask.Wait();
			// Policy cpuPolicy = cpuPolicyTask.Result;
#endif

			return gpuPolicy;

		}

		private IDecisionNode FitPolicyToHero(HeroType heroType, List<Episode> episodes, Policy policy) {
			DecisionNodeAccuracy rootAccuracy = CalculateDecisionTree(episodes);
			IDecisionNode root = rootAccuracy.Root;
			Console.WriteLine(string.Format(
				"GPU: Policy [{0}] ({1} leaves from {2} episodes) accuracy: {3}",
				heroType == HeroType.None ? "All" : heroType.ToString(),
				CountLeaves(root),
				episodes.Count,
				rootAccuracy.Accuracy));

			return root;
		}

		private DecisionNodeAccuracy CalculateDecisionTree(List<Episode> episodes) {
			List<DataPoint> dataPoints = EpisodesToDataPoints(episodes);
			using (DecisionLearner decisionLearner = new DecisionLearner(_cudaManager, dataPoints)) {
				DataNodeAccuracy rootAccuracy = decisionLearner.FitDecisionTree();
				double gpuAccuracy = rootAccuracy.Accuracy;

				IDecisionNode decisionTree = NodeToDecisionTree(rootAccuracy.Node);

#if DEBUGCUDA
				SanityCheckDecisionTree(rootAccuracy.Node, decisionTree, gpuAccuracy, episodes);
#endif

				return new DecisionNodeAccuracy {
					Root = decisionTree,
					Accuracy = gpuAccuracy,
				};
			}
		}

		private void SanityCheckDecisionTree(IDataNode root, IDecisionNode decisionTree, double gpuAccuracy, List<Episode> episodes) {
			double gpuTotalWeight = Leaves(root).Sum(x => x.ClassDistribution.Sum());
			double cpuTotalWeight = episodes.Sum(x => x.Weight);
			if (Math.Abs(gpuTotalWeight - cpuTotalWeight) > 1.0) {
				throw new InvalidOperationException("Weights don't match");
			}

			double cpuAccuracy = CalculateAccuracy(decisionTree, episodes);
			if (Math.Abs(gpuAccuracy - cpuAccuracy) > 0.01) {
				Console.WriteLine(string.Format("Inaccurate: CPU {0:F3} vs GPU {1:F3}", cpuAccuracy, gpuAccuracy));
			}
		}

		private static IDecisionNode NodeToDecisionTree(IDataNode node) {
			if (node is DataLeaf) {
				DataLeaf leaf = (DataLeaf)node;
				Tactic tactic = EnumUtils.GetEnumValues<Tactic>().MaxBy(t => leaf.ClassDistribution[(int)t]);
				return new DecisionLeaf(tactic);
			} else if (node is AttributeSplit) {
				AttributeSplit attributeSplit = (AttributeSplit)node;
				return new DecisionNode {
					Partitioner = new ContinuousPartitioner((ContinuousAxis)attributeSplit.Axis, attributeSplit.SplitValue),
					Left = NodeToDecisionTree(attributeSplit.Left),
					Right = NodeToDecisionTree(attributeSplit.Right),
				};
			} else if (node is CategoricalSplit) {
				CategoricalSplit categoricalSplit = (CategoricalSplit)node;
				return new DecisionNode {
					Partitioner = new CategoricalPartitioner(
						(CategoricalAxis)categoricalSplit.Axis,
						PolicyHelper.BitsToCategories((CategoricalAxis)categoricalSplit.Axis, categoricalSplit.Categories).ToArray()),
					Left = NodeToDecisionTree(categoricalSplit.Left),
					Right = NodeToDecisionTree(categoricalSplit.Right),
				};
			} else {
				throw new ArgumentException("Unknown node type: " + node);
			}
		}

		private static double CalculateAccuracy(IDecisionNode root, List<Episode> episodes) {
			double correctWeight = episodes.AsParallel().Sum(ep => root.Evaluate(ep.World, ep.Hero) == ep.Tactic ? ep.Weight : 0.0);
			double totalWeight = episodes.Sum(ep => ep.Weight);
			return correctWeight / totalWeight;
		}

		private static List<DataLeaf> Leaves(IDataNode root) {
			List<DataLeaf> leaves = new List<DataLeaf>();
			AppendLeaves(root, leaves);
			return leaves;
		}

		private static void AppendLeaves(IDataNode node, List<DataLeaf> leaves) {
			if (node is DataLeaf) {
				leaves.Add((DataLeaf)node);
			} else if (node is IDataSplit) {
				IDataSplit split = (IDataSplit)node;
				AppendLeaves(split.Left, leaves);
				AppendLeaves(split.Right, leaves);
			} else {
				throw new ArgumentException("Unknown node type: " + node);
			}
		}

		private static List<DataPoint> EpisodesToDataPoints(List<Episode> episodes) {
			return episodes.AsParallel().Select(episode => new DataPoint {
				Episode = episode,
				Class = (byte)episode.Tactic,
				Weight = (float)episode.Weight,
				Attributes =
					EnumUtils.GetEnumValues<ContinuousAxis>()
					.Select(axis => (float)ContinuousAxisEvaluator.Evaluate(axis, episode.World, episode.Hero))
					.ToArray(),
				Categories =
					EnumUtils.GetEnumValues<CategoricalAxis>()
					.Select(axis => PolicyHelper.CategoriesToBits(CategoricalAxisEvaluator.Evaluate(axis, episode.World, episode.Hero)))
					.ToArray(),
			}).ToList();
		}

		private static int CountLeaves(IDecisionNode node) {
			if (node is DecisionLeaf) {
				return 1;
			} else if (node is DecisionNode) {
				DecisionNode decisionNode = (DecisionNode)node;
				return CountLeaves(decisionNode.Left) + CountLeaves(decisionNode.Right);
			} else {
				throw new ArgumentException("Unknown node type: " + node);
			}
		}

		public void Dispose() {
			throw new NotImplementedException();
		}

		private class DataPoint : IDataPoint {
			public Episode Episode;

			public float[] Attributes { get; set; }
			public uint[] Categories { get; set; }
			public byte Class { get; set; }
			public float Weight { get; set; }
		}

		private class DecisionNodeAccuracy {
			public IDecisionNode Root;
			public double Accuracy;
		}
	}
}
