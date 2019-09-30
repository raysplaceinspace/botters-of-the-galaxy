using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BottersOTG.Training.DecisionLearning.Model;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Telogis.RouteCloud.GPUManagement;

namespace BottersOTG.Training.DecisionLearning.AttributeSplitting {
	class AttributeSplitter : IDisposable {
		public const int NumSmallBlocks = 128;
		public const int LargeBlock = GPUConstants.LargeBlock;

		public readonly DecisionLearnerContext Context;

		private CudaArray<GPUAttributeDataPoint> _sortedDataPointsPerAxis;
		private CudaArray<float> _sortKeysPerAxis;
		private CudaArray<float> _cumulativeFrequenciesPerAxis;

		public KernelManager Kernels {
			get { return Context.Kernels; }
		}

		public AttributeSplitter(DecisionLearnerContext context) {
			Context = context;
		}

		public void Initialize() {
			_sortedDataPointsPerAxis = new CudaArray<GPUAttributeDataPoint>(Context.DataPoints.Count * Context.NumAttributeAxes);
			_sortKeysPerAxis = new CudaArray<float>(Context.DataPoints.Count * Context.NumAttributeAxes);
			_cumulativeFrequenciesPerAxis = new CudaArray<float>(Context.DataPoints.Count * Context.NumAttributeAxes * GPUConstants.MaxClasses);
		}

		public void Dispose() {
			_sortKeysPerAxis?.Dispose();
			_sortedDataPointsPerAxis?.Dispose();
			_cumulativeFrequenciesPerAxis?.Dispose();
		}

		public int FindOptimalSplit(CudaArray<GPUSplit> splits, int writeSplitsStart) {
			Kernels["spaCopyDataPointsPerAxis"]
				.Arguments(_sortedDataPointsPerAxis, _sortKeysPerAxis)
				.Execute(GPUConstants.NumThreadsPerBlock * NumSmallBlocks, GPUConstants.NumThreadsPerBlock);

			Kernels["spaSortDataPointsPerAxis"]
				.Arguments(_sortedDataPointsPerAxis, _sortKeysPerAxis)
				.Execute(GPUConstants.NumThreadsPerBlock * NumSmallBlocks, GPUConstants.NumThreadsPerBlock);

			Kernels["spaAccumulateFrequenciesPerAxis"]
				.Arguments(_sortedDataPointsPerAxis, _cumulativeFrequenciesPerAxis)
				.Execute(GPUConstants.NumThreadsPerBlock * NumSmallBlocks, GPUConstants.NumThreadsPerBlock);
#if DEBUGCUDA
			SanityCheckCumulativeFrequencies();
#endif

			Kernels["spaBestSplitPerAxis"]
				.Arguments(_sortedDataPointsPerAxis, _cumulativeFrequenciesPerAxis, splits, writeSplitsStart)
				.Execute(GPUConstants.NumThreadsPerBlock * NumSmallBlocks, GPUConstants.NumThreadsPerBlock);

			int numSplitsAdded = Context.NumAttributeAxes;
			return numSplitsAdded;
		}

		public void ApplyOptimalSplit(CudaArray<GPUSplit> bestSplits) {
			Kernels["spaApplyOptimalSplit"]
				.Arguments(_sortedDataPointsPerAxis, bestSplits)
				.Execute(GPUConstants.NumThreadsPerBlock * NumSmallBlocks);

#if DEBUGCUDA
			SanityCheckSplit(bestSplits);
#endif
		}

		private void SanityCheckCumulativeFrequencies() {
			GPUDecisionLearnerContext context = Context.ContextBuffer.Read()[0];
			GPUAttributeDataPoint[] attributePoints = _sortedDataPointsPerAxis.Read();
			float[] cumulativeFrequenciesPerAxisArray = _cumulativeFrequenciesPerAxis.Read();
			for (int openNodeIndex = 0; openNodeIndex < context.NumOpenNodes; ++openNodeIndex) {
				int nodeId = Context.OpenNodeIds[openNodeIndex];
				if (nodeId == -1) {
					continue;
				}

				GPUNode node = Context.Nodes[nodeId];
				for (int axisId = 0; axisId < Context.NumAttributeAxes; ++axisId) {
					double[] cumulativeFrequencies = new double[GPUConstants.MaxClasses];
					for (int i = 0; i < node.RangeLength; ++i) {
						int index = node.RangeStart + i;
						GPUAttributeDataPoint attributePoint = attributePoints[axisId * Context.DataPoints.Count + index];
						cumulativeFrequencies[attributePoint.Class] += attributePoint.Weight;
						for (int classId = 0; classId < GPUConstants.MaxClasses; ++classId) {
							double cumulativeFrequency = cumulativeFrequenciesPerAxisArray[
								axisId * Context.DataPoints.Count * GPUConstants.MaxClasses +
								index * GPUConstants.MaxClasses +
								classId];
							Assert.AreEqual(cumulativeFrequencies[classId], cumulativeFrequency, 1.0);
						}
					}
				}
			}
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
				if (bestSplit.SplitType != GPUConstants.SplitType_Attribute) {
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
					bool goRight = dataPoint.Attributes[bestSplit.Axis] >= bestSplit.SplitAttribute;
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
