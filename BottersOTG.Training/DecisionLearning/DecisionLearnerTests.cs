using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BottersOTG.Intelligence.Decisions;
using BottersOTG.Training.DecisionLearning.Model;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Telogis.RouteCloud.GPUManagement;
using Utils;

namespace BottersOTG.Training.DecisionLearning {
	[TestClass]
	public class DecisionLearnerTests {
		[TestMethod]
		public void TestEnoughTacticMemory() {
			Assert.IsTrue(GPUConstants.MaxClasses >= EnumUtils.GetEnumValues<Tactic>().Count());
			Assert.IsTrue(GPUConstants.MaxCategoricalAxes >= EnumUtils.GetEnumValues<CategoricalAxis>().Count());
			Assert.IsTrue(GPUConstants.MaxAttributeAxes >= EnumUtils.GetEnumValues<ContinuousAxis>().Count());
		}

		[TestMethod]
		public void TestConstants() {
			Assert.IsTrue(GPUConstants.LargeBlock % GPUConstants.MaxClasses == 0);
			Assert.IsTrue(GPUConstants.MaxSplits >= GPUConstants.MaxAttributeAxes + GPUConstants.MaxCategoricalAxes);
			Assert.IsTrue(GPUConstants.MaxNodesAtSingleLevel >= 1 << (GPUConstants.MaxLevels - 1));
			Assert.IsTrue(GPUConstants.MaxTotalNodes >= (1 << GPUConstants.MaxLevels) - 1);
		}

		[TestMethod]
		public void TestEnumOrder() {
			Assert.IsTrue(EnumUtils.GetEnumValues<CategoricalAxis>().Select((axis, index) => (int)axis == index).All(x => x));
			Assert.IsTrue(EnumUtils.GetEnumValues<ContinuousAxis>().Select((axis, index) => (int)axis == index).All(x => x));
		}

		[TestMethod]
		public void TestCategoricalEnumConversion() {
			// Assert this doesn't crash
			EnumUtils.GetEnumValues<CategoricalAxis>().Select(axis => PolicyHelper.BitsToCategories(axis, 1));
		}


		[TestMethod]
		public void TestSum() {
			int[] numbers = Enumerable.Range(999, 10000).ToArray();
			using (CudaManager cudaManager = Provider.CudaManagerPool.GetCudaManagerForThread(Provider.Logger))
			using (CudaArray<int> numbersBuffer = numbers)
			using (CudaArray<int> outputBuffer = new[] { 0 }) {
				KernelManager kernels = new KernelManager(cudaManager);

				for (int nBlocks = 1; nBlocks <= 1024; nBlocks *= 2) {
					for (int threadsPerBlock = 1; threadsPerBlock <= 1024; threadsPerBlock *= 2) {
						kernels["setIntKernel"].Arguments(outputBuffer, 0).ExecuteTask();
						Assert.AreEqual(0, outputBuffer.Read()[0]);
						kernels["sumToOutputKernel"].Arguments(numbersBuffer, numbers.Length, outputBuffer, SharedBuffer.Ints(threadsPerBlock)).Execute(nBlocks * threadsPerBlock, threadsPerBlock);
						Assert.AreEqual(numbers.Sum(), outputBuffer.Read()[0]);
					}
				}
			}
		}

		[TestMethod]
		public void TestSimpleAttributeSplit() {
			List<DataPoint> dataPoints = new List<DataPoint> {
				new DataPoint {
					Class = 0,
					Attributes = new float[] { 123 },
					Categories = new uint[] { },
					Weight = 1.0f,
				},
				new DataPoint {
					Class = 1,
					Attributes = new float[] { 456 },
					Categories = new uint[] { },
					Weight = 1.0f,
				},
			};

			using (CudaManager cudaManager = Provider.CudaManagerPool.GetCudaManagerForThread(Provider.Logger))
			using (DecisionLearner decisionLearner = new DecisionLearner(cudaManager, dataPoints)) {
				IDataNode root = decisionLearner.FitDecisionTree().Node;

				AttributeSplit attributeSplit = (AttributeSplit)root;
				Assert.AreEqual(0, attributeSplit.Axis);
				Assert.AreEqual(456, attributeSplit.SplitValue);

				DataLeaf left = (DataLeaf)attributeSplit.Left;
				Assert.AreEqual(1, left.ClassDistribution[0]);
				Assert.AreEqual(0, left.ClassDistribution[1]);

				DataLeaf right = (DataLeaf)attributeSplit.Right;
				Assert.AreEqual(0, right.ClassDistribution[0]);
				Assert.AreEqual(1, right.ClassDistribution[1]);

				Assert.AreEqual(1.0, Accuracy(root, dataPoints));
			}
		}

		[TestMethod]
		public void TestTwoAttributeAxes() {
			List<DataPoint> dataPoints = new List<DataPoint> {
				new DataPoint {
					Class = 1,
					Attributes = new float[] { 123, 987 },
					Categories = new uint[] { },
					Weight = 1.0f,
				},
				new DataPoint {
					Class = 0,
					Attributes = new float[] { 123, 876 },
					Categories = new uint[] { },
					Weight = 1.0f,
				},
				new DataPoint {
					Class = 1,
					Attributes = new float[] { 456, 765 },
					Categories = new uint[] { },
					Weight = 1.0f,
				},
				new DataPoint {
					Class = 1,
					Attributes = new float[] { 456, 543 },
					Categories = new uint[] { },
					Weight = 1.0f,
				},
			};

			using (CudaManager cudaManager = Provider.CudaManagerPool.GetCudaManagerForThread(Provider.Logger))
			using (DecisionLearner decisionLearner = new DecisionLearner(cudaManager, dataPoints)) {
				IDataNode root = decisionLearner.FitDecisionTree().Node;
				Assert.AreEqual(1.0, Accuracy(root, dataPoints));
			}
		}

		[TestMethod]
		public void TestSimpleCategoricalSplit() {
			List<DataPoint> dataPoints = new List<DataPoint> {
				new DataPoint {
					Class = 0,
					Attributes = new float[] { },
					Categories = new uint[] { 0 },
					Weight = 1.0f,
				},
				new DataPoint {
					Class = 1,
					Attributes = new float[] { },
					Categories = new uint[] { 1 },
					Weight = 1.0f,
				},
			};

			using (CudaManager cudaManager = Provider.CudaManagerPool.GetCudaManagerForThread(Provider.Logger))
			using (DecisionLearner decisionLearner = new DecisionLearner(cudaManager, dataPoints)) {
				IDataNode root = decisionLearner.FitDecisionTree().Node;

				CategoricalSplit categoricalSplit = (CategoricalSplit)root;
				Assert.AreEqual(0, categoricalSplit.Axis);
				Assert.AreEqual((uint)1, categoricalSplit.Categories);

				DataLeaf left = (DataLeaf)categoricalSplit.Left;
				Assert.AreEqual(1, left.ClassDistribution[0]);
				Assert.AreEqual(0, left.ClassDistribution[1]);

				DataLeaf right = (DataLeaf)categoricalSplit.Right;
				Assert.AreEqual(0, right.ClassDistribution[0]);
				Assert.AreEqual(1, right.ClassDistribution[1]);

				Assert.AreEqual(1.0, Accuracy(root, dataPoints));
			}
		}

		[TestMethod]
		public void TestCategoryPlusAttributeSplit() {
			List<DataPoint> dataPoints = new List<DataPoint> {
				new DataPoint {
					Class = 1,
					Attributes = new float[] { 99, 987 },
					Categories = new uint[] { 0x3, 0x1 },
					Weight = 1.0f,
				},
				new DataPoint {
					Class = 0,
					Attributes = new float[] { 99, 987 },
					Categories = new uint[] { 0x3, 0x1 | 0x2 },
					Weight = 1.0f,
				},
				new DataPoint {
					Class = 1,
					Attributes = new float[] { 99, 765 },
					Categories = new uint[] { 0x3, 0x1 | 0x2 },
					Weight = 1.0f,
				},
				new DataPoint {
					Class = 1,
					Attributes = new float[] { 99, 765 },
					Categories = new uint[] { 0x3, 0x1 },
					Weight = 1.0f,
				},

			};

			using (CudaManager cudaManager = Provider.CudaManagerPool.GetCudaManagerForThread(Provider.Logger))
			using (DecisionLearner decisionLearner = new DecisionLearner(cudaManager, dataPoints)) {
				IDataNode root = decisionLearner.FitDecisionTree().Node;

				Assert.AreEqual(1.0, Accuracy(root, dataPoints));
			}
		}

		private class DataPoint : IDataPoint {
			public float[] Attributes { get; set; }
			public uint[] Categories { get; set; }
			public byte Class { get; set; }

			public float Weight { get; set; }
		}

		private static double Accuracy(IDataNode root, IEnumerable<DataPoint> dataPoints) {
			return dataPoints.Average(dataPoint => {
				DataLeaf leaf = Resolve(root, dataPoint);
				return (double)leaf.ClassDistribution[dataPoint.Class] / leaf.ClassDistribution.Sum();
			});
		}

		private static DataLeaf Resolve(IDataNode node, DataPoint newDataPoint) {
			if (node is DataLeaf) {
				return (DataLeaf)node;
			} else if (node is AttributeSplit) {
				AttributeSplit attributeSplit = (AttributeSplit)node;
				if (newDataPoint.Attributes[attributeSplit.Axis] >= attributeSplit.SplitValue) {
					return Resolve(attributeSplit.Right, newDataPoint);
				} else {
					return Resolve(attributeSplit.Left, newDataPoint);
				}
			} else if (node is CategoricalSplit) {
				CategoricalSplit categoricalSplit = (CategoricalSplit)node;
				if ((newDataPoint.Categories[categoricalSplit.Axis] & categoricalSplit.Categories) != 0) {
					return Resolve(categoricalSplit.Right, newDataPoint);
				} else {
					return Resolve(categoricalSplit.Left, newDataPoint);
				}
			} else {
				throw new ArgumentException("Unknown node type: " + node);
			}
		}
	}
}
