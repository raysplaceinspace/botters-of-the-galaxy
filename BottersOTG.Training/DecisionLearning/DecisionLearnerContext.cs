using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Telogis.RouteCloud.GPUManagement;
using BottersOTG.Training.DecisionLearning.Model;

namespace BottersOTG.Training.DecisionLearning {
	unsafe class DecisionLearnerContext {
		public CudaManager CudaManager { get; private set; }
		public KernelManager Kernels { get; private set; }

		public IList<IDataPoint> DataPoints { get; private set; }
		public float TotalWeight;
		public int NumAttributeAxes { get; private set; }
		public int NumCategoricalAxes { get; private set; }

		public CudaArray<GPUDecisionLearnerContext> ContextBuffer;
		public CudaArray<GPUDataPoint> DataPointBuffer;
		public CudaArray<int> DataPointIds;

		public CudaArray<GPUNode> Nodes;
		public CudaArray<int> OpenNodeIds;
		public CudaArray<int> NextOpenNodeIds;

		public DecisionLearnerContext(CudaManager cudaManager, IEnumerable<IDataPoint> dataPoints) {
			CudaManager = cudaManager;
			DataPoints = dataPoints.ToList();
			TotalWeight = DataPoints.Sum(dp => dp.Weight);
			if (DataPoints.Count > 0) {
				NumAttributeAxes = DataPoints[0].Attributes.Length;
				NumCategoricalAxes = DataPoints[0].Categories.Length;
			}
		}

		public void Initialize() {
			if (NumAttributeAxes > GPUConstants.MaxAttributeAxes || NumCategoricalAxes > GPUConstants.MaxCategoricalAxes) {
				throw new InvalidOperationException("Too attribute axes");
			}
			ContextBuffer = new CudaArray<GPUDecisionLearnerContext>(1);
			DataPointBuffer = DataPointsToGpu(DataPoints);
			DataPointIds = new CudaArray<int>(DataPoints.Count);
			Nodes = new CudaArray<GPUNode>(GPUConstants.MaxTotalNodes);
			OpenNodeIds = new CudaArray<int>(GPUConstants.MaxNodesAtSingleLevel);
			NextOpenNodeIds = new CudaArray<int>(GPUConstants.MaxNodesAtSingleLevel);

			Kernels = new KernelManager(CudaManager) {
				BlockSize = GPUConstants.NumThreadsPerBlock,
				PrefixArguments = new[] { ContextBuffer },
			};

			Kernels["dlInitContext"].Arguments(
				DataPointBuffer,
				DataPoints.Count,
				TotalWeight,
				NumAttributeAxes,
				NumCategoricalAxes,
				DataPointIds,
				Nodes,
				OpenNodeIds,
				NextOpenNodeIds,
				(int)OpenNodeIds.Size).ExecuteTask();
		}

		public void Dispose() {
			Nodes?.Dispose();
			NextOpenNodeIds?.Dispose();
			OpenNodeIds?.Dispose();
			DataPointIds?.Dispose();
			DataPointBuffer?.Dispose();
			ContextBuffer?.Dispose();
		}

		private static GPUDataPoint[] DataPointsToGpu(IEnumerable<IDataPoint> dataPoints) {
			return dataPoints.Select(dataPoint => {
				GPUDataPoint gpuDataPoint = new GPUDataPoint {
					Class = dataPoint.Class,
					Weight = dataPoint.Weight,
				};
				for (int i = 0; i < dataPoint.Attributes.Length; ++i) {
					gpuDataPoint.AllAttributes[i] = dataPoint.Attributes[i];
				}
				for (int i = 0; i < dataPoint.Categories.Length; ++i) {
					gpuDataPoint.AllCategories[i] = dataPoint.Categories[i];
				}
				return gpuDataPoint;
			}).ToArray();
		}

	}
}
