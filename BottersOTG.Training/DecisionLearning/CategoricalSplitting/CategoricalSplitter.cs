using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BottersOTG.Training.DecisionLearning.Model;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Telogis.RouteCloud.GPUManagement;

namespace BottersOTG.Training.DecisionLearning.CategoricalSplitting {
	class CategoricalSplitter : IDisposable {
		public const int NumSmallBlocks = 128;
		public const int LargeBlock = GPUConstants.LargeBlock;

		public readonly DecisionLearnerContext Context;

		private CudaArray<GPUCategoricalDataPoint> _dataPointsPerAxis;
		private CudaArray<float> _classFrequenciesPerCategory;
		private CudaArray<byte> _sortKeys;

		public KernelManager Kernels {
			get { return Context.Kernels; }
		}

		public CategoricalSplitter(DecisionLearnerContext context) {
			Context = context;
		}

		public void Initialize() {
			_dataPointsPerAxis = new CudaArray<GPUCategoricalDataPoint>(Context.DataPoints.Count * GPUConstants.MaxCategoricalAxes);
			_classFrequenciesPerCategory = new CudaArray<float>(GPUConstants.MaxClasses * GPUConstants.MaxCategoricalAxes);
			_sortKeys = new CudaArray<byte>(Context.DataPoints.Count);
		}

		public void Dispose() {
			_sortKeys?.Dispose();
			_dataPointsPerAxis?.Dispose();
			_classFrequenciesPerCategory?.Dispose();
		}

		public int FindOptimalSplit(CudaArray<GPUSplit> bestSplits, int writeSplitsStart) {
			Kernels["spcCopyDataPointsPerAxis"]
				.Arguments(_dataPointsPerAxis)
				.Execute(NumSmallBlocks * GPUConstants.NumThreadsPerBlock, GPUConstants.NumThreadsPerBlock);

			Kernels["spcBestCategoricalSplitPerAxis"]
				.Arguments(_dataPointsPerAxis, bestSplits, writeSplitsStart)
				.Execute(NumSmallBlocks * GPUConstants.NumThreadsPerBlock, GPUConstants.NumThreadsPerBlock);
			return Context.NumCategoricalAxes;
		}

		public void ApplyOptimalSplit(CudaArray<GPUSplit> bestSplits) {
			Kernels["spcApplyOptimalSplit"]
				.Arguments(_dataPointsPerAxis, bestSplits, _sortKeys)
				.Execute(GPUConstants.NumThreadsPerBlock * NumSmallBlocks);

#if DEBUGCUDA
			SanityCheckSplit(bestSplits);
#endif
		}

		private void SanityCheckSplit(CudaArray<GPUSplit> bestSplits) {
			GPUSplit[] bestSplitsArray = bestSplits.Read();
			GPUDecisionLearnerContext context = Context.ContextBuffer.Read()[0];
			int[] dataPointIds = Context.DataPointIds.Read();
			for (int openNodeIndex = 0; openNodeIndex < context.NumOpenNodes; ++openNodeIndex) {
				int nodeId = Context.OpenNodeIds[openNodeIndex];
				if (nodeId == -1) {
					continue;
				}

				GPUSplit bestSplit = bestSplitsArray[openNodeIndex];
				if (bestSplit.SplitType != GPUConstants.SplitType_Categorical) {
					continue;
				}

				GPUNode parentNode = Context.Nodes[nodeId];
				GPUNode leftNode = Context.Nodes[parentNode.LeftChild];
				GPUNode rightNode = Context.Nodes[parentNode.RightChild];
				Assert.AreEqual(parentNode.RangeStart, leftNode.RangeStart);
				Assert.AreEqual(leftNode.RangeStart + leftNode.RangeLength, rightNode.RangeStart);
				Assert.AreEqual(parentNode.RangeLength, leftNode.RangeLength + rightNode.RangeLength);

				for (int i = 0; i < parentNode.RangeLength; ++i) {
					int index = parentNode.RangeStart + i;
					IDataPoint dataPoint = Context.DataPoints[dataPointIds[index]];
					bool goRight = (dataPoint.Categories[bestSplit.Axis] & bestSplit.SplitCategories) != 0;
					if (goRight) {
						Assert.IsTrue(rightNode.RangeStart <= index && index <= rightNode.RangeStart + rightNode.RangeLength);
					} else {
						Assert.IsTrue(leftNode.RangeStart <= index && index <= leftNode.RangeStart + leftNode.RangeLength);
					}
				}
			}
		}
	}
}
