using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BottersOTG.Intelligence.Decisions;
using BottersOTG.Training.DecisionLearning.AttributeSplitting;
using BottersOTG.Training.DecisionLearning.CategoricalSplitting;
using BottersOTG.Training.DecisionLearning.Model;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Telogis.RouteCloud.GPUManagement;
using Utils;

namespace BottersOTG.Training.DecisionLearning {
	unsafe class DecisionLearner : IDisposable {
		public const int NumSmallBlocks = 128;
		public const int LargeBlock = GPUConstants.LargeBlock;

		public DecisionLearnerContext Context { get; private set; }

		private bool _initialized;

		private readonly AttributeSplitter _attributeSplitter;
		private readonly CategoricalSplitter _categoricalSplitter;

		private CudaArray<GPUSplit> _allSplits;
		private CudaArray<GPUSplit> _bestSplits;

		public KernelManager Kernels {
			get { return Context.Kernels; }
		}

		public DecisionLearner(CudaManager cudaManager, IEnumerable<IDataPoint> dataPoints) {
			Context = new DecisionLearnerContext(cudaManager, dataPoints);
			_attributeSplitter = new AttributeSplitter(Context);
			_categoricalSplitter = new CategoricalSplitter(Context);
		}

		public void Initialize() {
			if (_initialized) {
				return;
			} else {
				_initialized = true;
			}
			Context.Initialize();

			_allSplits = new CudaArray<GPUSplit>(GPUConstants.MaxSplits * Context.OpenNodeIds.Size);
			_bestSplits = new CudaArray<GPUSplit>(Context.OpenNodeIds.Size);

			_attributeSplitter.Initialize();
			_categoricalSplitter.Initialize();

			Context.CudaManager.Context.Synchronize();
		}

		public void Dispose() {
			_bestSplits?.Dispose();
			_allSplits?.Dispose();
			_attributeSplitter?.Dispose();
			_categoricalSplitter?.Dispose();
			Context?.Dispose();
		}

		public DataNodeAccuracy FitDecisionTree() {
			Initialize();

			Kernels["dlInitDataPoints"].Execute(Context.DataPoints.Count);
			Kernels["dlInitialNode"].ExecuteTask();
			for (int i = 0; i < GPUConstants.MaxLevels; ++i) {
				Kernels["dlLeafOpenNodes"]
					.Execute(NumSmallBlocks * GPUConstants.NumThreadsPerBlock, GPUConstants.NumThreadsPerBlock);
				bool isFinalLevel = i == GPUConstants.MaxLevels - 1;
				if (isFinalLevel) {
					break; // Can't split anymore
				}

				int numSplits = 0;

				numSplits += _attributeSplitter.FindOptimalSplit(_allSplits, numSplits);
				numSplits += _categoricalSplitter.FindOptimalSplit(_allSplits, numSplits);

				Kernels["dlFindOptimalSplitPerNode"]
					.Arguments(_allSplits, numSplits, _bestSplits)
					.Execute(NumSmallBlocks * GPUConstants.NumThreadsPerBlock, GPUConstants.NumThreadsPerBlock);

				_attributeSplitter.ApplyOptimalSplit(_bestSplits);
				_categoricalSplitter.ApplyOptimalSplit(_bestSplits);

				Kernels["dlNextLevel"]
					.Execute(NumSmallBlocks * GPUConstants.NumThreadsPerBlock, GPUConstants.NumThreadsPerBlock);
			}

			GPUNode[] gpuNodes = Context.Nodes.Read();
#if DEBUGCUDA
			SanityCheckGPUNodes(gpuNodes);
#endif
			IDataNode unpruned = ReadDecisionTree(gpuNodes, 0);
			DataNodeAccuracy root = PruneDecisionTree(unpruned);
			return root;
		}

		private DataNodeAccuracy PruneDecisionTree(IDataNode node) {
			if (node is DataLeaf) {
				return new DataNodeAccuracy {
					Node = node,
					CorrectWeight = node.ClassDistribution.Max(),
					TotalWeight = node.ClassDistribution.Sum(),
				};
			} else if (node is IDataSplit) {
				IDataSplit split = (IDataSplit)node;
				DataNodeAccuracy leftAccuracy = PruneDecisionTree(split.Left);
				DataNodeAccuracy rightAccuracy = PruneDecisionTree(split.Right);

				float splitCorrect = leftAccuracy.CorrectWeight + rightAccuracy.CorrectWeight;
				float splitTotal = leftAccuracy.TotalWeight + rightAccuracy.TotalWeight;

				float leafCorrect = node.ClassDistribution.Max();
				float leafTotal = node.ClassDistribution.Sum();

#if DEBUGCUDA
				Assert.AreEqual(leafTotal, splitTotal, 1.0f);
#endif

				float accuracyIncrease = splitCorrect - leafCorrect;
				float requiredIncrease = Context.TotalWeight * GPUConstants.RequiredImprovementToSplit;
				if (accuracyIncrease < requiredIncrease) {
					// Split not justified, revert back to leaf at this level
					return new DataNodeAccuracy {
						Node = new DataLeaf { ClassDistribution = node.ClassDistribution },
						CorrectWeight = leafCorrect,
						TotalWeight = leafTotal,
					};
				} else {
					// Take the split with the pruned nodes
					if (split is AttributeSplit) {
						AttributeSplit attributeSplit = (AttributeSplit)split;
						attributeSplit.Left = leftAccuracy.Node;
						attributeSplit.Right = rightAccuracy.Node;
					} else if (split is CategoricalSplit) {
						CategoricalSplit categoricalSplit = (CategoricalSplit)split;
						categoricalSplit.Left = leftAccuracy.Node;
						categoricalSplit.Right = rightAccuracy.Node;
					} else {
						throw new ArgumentException("Unknown split type: " + split);
					}
					return new DataNodeAccuracy {
						Node = split,
						CorrectWeight = splitCorrect,
						TotalWeight = splitTotal,
					};
				}
			} else {
				throw new ArgumentException("Unknown node type: " + node);
			}
		}

		private void SanityCheckGPUNodes(GPUNode[] gpuNodes) {
			bool[] covered = new bool[Context.DataPoints.Count];
			List<GPUNode> leaves = new List<GPUNode>();
			AppendLeaves(gpuNodes, 0, leaves);

			foreach (GPUNode leaf in leaves) {
				for (int i = 0; i < leaf.RangeLength; ++i) {
					int index = leaf.RangeStart + i;
					if (covered[index]) {
						throw new InvalidOperationException("Already covered: " + index);
					} else {
						covered[index] = true;
					}
				}
			}
		}

		private void AppendLeaves(GPUNode[] gpuNodes, int nodeIndex, List<GPUNode> leaves) {
			GPUNode node = gpuNodes[nodeIndex];
			switch (node.SplitType) {
				case GPUConstants.SplitType_Null:
					throw new InvalidOperationException("Null node in tree");
				case GPUConstants.SplitType_None:
					leaves.Add(node);
					break;
				case GPUConstants.SplitType_Attribute:
				case GPUConstants.SplitType_Categorical:
					AppendLeaves(gpuNodes, node.LeftChild, leaves);
					AppendLeaves(gpuNodes, node.RightChild, leaves);
					break;
				default:
					throw new InvalidOperationException("Unknown split type: " + node.SplitType);
			}
		}

		private IDataNode ReadDecisionTree(GPUNode[] nodes, int nodeIndex) {
			GPUNode node = nodes[nodeIndex];
			switch (node.SplitType) {
				case GPUConstants.SplitType_Null:
					throw new InvalidOperationException("Null node in tree");
				case GPUConstants.SplitType_None:
					return new DataLeaf {
						ClassDistribution = ReadClassDistribution(node.ClassDistribution),
					};
				case GPUConstants.SplitType_Attribute:
					return new AttributeSplit {
						Axis = node.SplitAxis,
						SplitValue = node.SplitAttribute,
						Left = ReadDecisionTree(nodes, node.LeftChild),
						Right = ReadDecisionTree(nodes, node.RightChild),
						ClassDistribution = ReadClassDistribution(node.ClassDistribution),
					};
				case GPUConstants.SplitType_Categorical:
					return new CategoricalSplit {
						Axis = node.SplitAxis,
						Categories = node.SplitCategories,
						Left = ReadDecisionTree(nodes, node.LeftChild),
						Right = ReadDecisionTree(nodes, node.RightChild),
						ClassDistribution = ReadClassDistribution(node.ClassDistribution),
					};
				default:
					throw new InvalidOperationException("Unknown split type: " + node.SplitType);
			}
		}

		private float[] ReadClassDistribution(float* gpuClassDistribution) {
			float[] result = new float[GPUConstants.MaxClasses];
			for (int i = 0; i < GPUConstants.MaxClasses; ++i) {
				result[i] = gpuClassDistribution[i];
			}
			return result;
		}
	}
}
