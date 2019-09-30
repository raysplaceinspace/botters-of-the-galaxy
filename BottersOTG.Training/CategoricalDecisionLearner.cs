using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BottersOTG.Intelligence.Decisions;
using Utils;

namespace BottersOTG.Training {
	public class CategoricalDecisionLearner {
		private readonly ConcurrentDictionary<Tuple<Episode, CategoricalAxis>, Enum[]> _evaluateCache = new ConcurrentDictionary<Tuple<Episode, CategoricalAxis>, Enum[]>();

		public IEnumerable<Func<PartitionScore>> Optimizers(List<Episode> episodes, double[] totalTacticWeights) {
			foreach (CategoricalAxis axis in EnumUtils.GetEnumValues<CategoricalAxis>()) {
				yield return () => SplitOnAxis(axis, episodes, totalTacticWeights);
			}
		}

		private PartitionScore SplitOnAxis(CategoricalAxis axis, List<Episode> episodes, double[] totalWeights) {
			// Calculate categories
			Dictionary<Enum, HashSet<Episode>> episodesByCategory =
				episodes
				.AsParallel()
				.Select(episode => EvaluateCategories(axis, episode))
				.SelectMany(x => x.Categories.Select(category => new {
					Category = category,
					Episode = x.Episode,
				}))
				.GroupBy(g => g.Category)
				.ToDictionary(g => g.Key, g => new HashSet<Episode>(g.Select(x => x.Episode)));

			List<PartitionScore> partitions = new List<PartitionScore>();

			HashSet<Episode> leftEpisodes = new HashSet<Episode>(episodes);
			List<Episode> rightEpisodes = new List<Episode>();
			HashSet<Enum> rightCategories = new HashSet<Enum>();
			double[] rightWeights = new double[TacticEntropy.NumTactics];

			while (leftEpisodes.Count > 0) {
				List<AdditionalCategory> additionalCategories =
					episodesByCategory
					.AsParallel()
					.Select(kvp => EvaluateAdditionalCategory(kvp.Key, kvp.Value, totalWeights, rightWeights))
					.ToList();
				AdditionalCategory bestAdditionalCategory = additionalCategories.MinByOrDefault(x => x.Entropy);
				if (bestAdditionalCategory == null) {
					// Everything left belongs to no categories - nothing to move
					break;
				}

				// Update left and right
				rightCategories.Add(bestAdditionalCategory.Category);
				leftEpisodes.ExceptWith(bestAdditionalCategory.AdditionalEpisodes);
				rightEpisodes.AddRange(bestAdditionalCategory.AdditionalEpisodes);
				rightWeights = bestAdditionalCategory.NewRightWeights;

				// Create snapshot of partition
				partitions.Add(new PartitionScore {
					Partitioner = new CategoricalPartitioner(axis, rightCategories.ToArray()),
					LeftEpisodes = leftEpisodes.ToList(),
					RightEpisodes = rightEpisodes.ToList(),
					Entropy = bestAdditionalCategory.Entropy,
				});

				// Remove additional episodes from other categories so we don't re-add them
				episodesByCategory.Remove(bestAdditionalCategory.Category);
				foreach (HashSet<Episode> otherEpisodes in episodesByCategory.Values) {
					otherEpisodes.ExceptWith(bestAdditionalCategory.AdditionalEpisodes);
				}
			}

			return partitions.MinByOrDefault(p => p.Entropy);
		}

		private static AdditionalCategory EvaluateAdditionalCategory(Enum category, IEnumerable<Episode> additionalEpisodes, double[] totalWeights, double[] rightWeights) {
			double[] additionalWeights = TacticEntropy.TacticFrequency(additionalEpisodes);
			double[] newRightWeights = new double[TacticEntropy.NumTactics];
			double[] newLeftWeights = new double[TacticEntropy.NumTactics];
			for (int i = 0; i < additionalWeights.Length; ++i) {
				newRightWeights[i] = rightWeights[i] + additionalWeights[i];
				newLeftWeights[i] = totalWeights[i] - newRightWeights[i];
			}

			return new AdditionalCategory {
				Category = category,
				AdditionalEpisodes = additionalEpisodes,
				NewRightWeights = newRightWeights,
				Entropy = TacticEntropy.Entropy(newLeftWeights, newRightWeights),
			};
		}

 		private CategorisedEpisode EvaluateCategories(CategoricalAxis axis, Episode episode) {
			var key = Tuple.Create(episode, axis);
			Enum[] categories;
			if (!_evaluateCache.TryGetValue(key, out categories)) {
				categories = CategoricalAxisEvaluator.Evaluate(axis, episode.World, episode.Hero).ToArray();
				_evaluateCache[key] = categories;
			}
 			return new CategorisedEpisode {
 				Episode = episode,
				Categories = categories,
 			};
 		}

		private class CategorisedEpisode {
			public Episode Episode;
			public Enum[] Categories;
		}

		private class AdditionalCategory {
			public Enum Category;
			public IEnumerable<Episode> AdditionalEpisodes;
			public double[] NewRightWeights;
			public double Entropy;
		}
	}
}
