using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BottersOTG.Intelligence.Decisions;
using Utils;

namespace BottersOTG.Training {
	public static class TacticEntropy {
		public static readonly int NumTactics = EnumUtils.GetEnumValues<Tactic>().Count();

		public static double Entropy(double accuracy) {
			if (accuracy == 0) {
				return 0;
			} else {
				return -accuracy * Math.Log(accuracy);
			}
		}

		public static double Entropy(double[] weights) {
			return Entropy(weights, weights.Sum());
		}

		public static double Entropy(double[] weights, double totalWeight) {
			double entropy = 0;
			for (int i = 0; i < weights.Length; ++i) {
				entropy += Entropy(weights[i] / totalWeight);
			}
			return entropy / weights.Length;
		}

		public static double Entropy(double[] weightsLeft, double[] weightsRight) {
			double weightLeft = weightsLeft.Sum();
			double weightRight = weightsRight.Sum();
			double weightTotal = weightLeft + weightRight;

			double entropyLeft = Entropy(weightsLeft, weightLeft);
			double entropyRight = Entropy(weightsRight, weightRight);
			double entropy = (entropyLeft * weightLeft / weightTotal) + (entropyRight * weightRight / weightTotal);

			return entropy;
		}

		public static double[] Subtract(double[] totalWeights, double[] tacticWeightsSide) {
			double[] result = new double[totalWeights.Length];
			for (int i = 0; i < totalWeights.Length; ++i) {
				result[i] = totalWeights[i] - tacticWeightsSide[i];
			}
			return result;
		}

		public static double[] TacticFrequency(IEnumerable<Episode> episodes) {
			double[] tacticWeights = new double[NumTactics];
			foreach (Episode episode in episodes) {
				tacticWeights[(int)episode.Tactic] += episode.Weight;
			}
			return tacticWeights;
		}
	}
}
