using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BottersOTG.Intelligence.Decisions;
using BottersOTG.Model;
using Utils;

namespace BottersOTG.Training {
	public class ContinuousDecisionLearner {
		private static readonly int NumTactics = EnumUtils.GetEnumValues<Tactic>().Count();

		private readonly ConcurrentDictionary<Tuple<Episode, ContinuousAxis>, double> _evaluateCache = new ConcurrentDictionary<Tuple<Episode, ContinuousAxis>, double>();

		public IEnumerable<Func<PartitionScore>> Optimizers(List<Episode> episodes, double[] totalTacticWeights) {
			foreach (ContinuousAxis axis in EnumUtils.GetEnumValues<ContinuousAxis>()) {
				yield return () => SplitOnAxis(axis, episodes, totalTacticWeights);
			}
		}

		private PartitionScore SplitOnAxis(ContinuousAxis axis, List<Episode> episodes, double[] totalWeights) {
			List<EvaluatedEpisode> line =
				episodes
				.AsParallel()
				.Select(episode => EvaluateEpisode(episode, axis))
				.OrderBy(x => x.Value)
				.ToList();

			int lastIndex = 0;

			double[] tacticWeightsLeft = new double[NumTactics];
			int bestSplitPosition = -1;
			double bestEntropy = double.MaxValue;
			for (int i = 1; i < line.Count; ++i) {
				bool sameAsPrevious = i > 0 && line[i - 1].Value == line[i].Value;
				if (sameAsPrevious) {
					// Cannot create a split part way through the same number
					continue;
				}

				while (lastIndex < i) {
					Episode episode = line[lastIndex++].Episode;
					tacticWeightsLeft[(int)episode.Tactic] += episode.Weight;
				}
				double[] tacticWeightsRight = TacticEntropy.Subtract(totalWeights, tacticWeightsLeft);

				double entropy = TacticEntropy.Entropy(tacticWeightsLeft, tacticWeightsRight);

				if (entropy < bestEntropy) {
					bestEntropy = entropy;
					bestSplitPosition = i;
				}
			}

			if (bestSplitPosition == -1) {
				// Unable to find split - all values must be the same
				return null;
			}

			return new PartitionScore {
				Partitioner = new ContinuousPartitioner(axis, line[bestSplitPosition].Value),
				LeftEpisodes = line.Take(bestSplitPosition).Select(x => x.Episode).ToList(),
				RightEpisodes = line.Skip(bestSplitPosition).Select(x => x.Episode).ToList(),
				Entropy = bestEntropy,
			};
		}

 		private EvaluatedEpisode EvaluateEpisode(Episode episode, ContinuousAxis axis) {
			var key = Tuple.Create(episode, axis);
			double value;
			if (!_evaluateCache.TryGetValue(key, out value)) {
				value = ContinuousAxisEvaluator.Evaluate(axis, episode.World, episode.Hero);
				_evaluateCache[key] = value;
			}
 			return new EvaluatedEpisode {
 				Episode = episode,
				Value = value,
 			};
 		}

		private class EvaluatedEpisode {
			public Episode Episode;
			public double Value;
		}
	}
}
