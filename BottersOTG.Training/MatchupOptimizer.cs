using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BottersOTG.Intelligence.Decisions;
using BottersOTG.Model;
using Utils;

namespace BottersOTG.Training {
	static class MatchupOptimizer {
		public static HeroChoices Optimize(IEnumerable<Rollout> tournament) {
			var primaryChoice =
				tournament
				.SelectMany(matchup => PairingsFromMatchup(matchup))
				.GroupBy(p => p.MyHero, (heroType, group) => new {
					HeroType = heroType,
					WinRate = group.Average(x => x.WinRate),
				})
				.MaxBy(x => x.WinRate);
			Console.WriteLine(string.Format("Primary: {0} ({1:F3})", primaryChoice.HeroType, primaryChoice.WinRate));

			var secondaryChoices =
				tournament
				.Where(matchupResult => matchupResult.Matchup.Team0.Contains(primaryChoice.HeroType))
				.SelectMany(matchup => PairingsFromMatchup(matchup))
				.Where(p => p.MyHero != primaryChoice.HeroType)
				.GroupBy(p => Tuple.Create(p.EnemyHero, p.MyHero), (tuple, group) => new {
					EnemyHero = tuple.Item1,
					MyHero = tuple.Item2,
					WinRate = group.Average(x => x.WinRate),
				})
				.GroupBy(x => x.EnemyHero, (enemyHero, group) => group.MaxBy(g => g.WinRate))
				.ToList();
			foreach (var secondaryChoice in secondaryChoices) {
				Console.WriteLine(string.Format(
					"Secondary: {0} -> {1} ({2:F3})",
					secondaryChoice.EnemyHero,
					secondaryChoice.MyHero,
					secondaryChoice.WinRate));
			}

			return new HeroChoices {
				PrimaryChoice = primaryChoice.HeroType,
				SecondaryChoices = secondaryChoices.ToDictionary(x => x.EnemyHero, x => x.MyHero),
			};
		}

		private static IEnumerable<HeroPairing> PairingsFromMatchup(Rollout matchupResult) {
			foreach (HeroType myHero in matchupResult.Matchup.Team0) {
				foreach (HeroType enemyHero in matchupResult.Matchup.Team1) {
					yield return new HeroPairing {
						MyHero = myHero,
						EnemyHero = enemyHero,
						WinRate = matchupResult.WinRate,
					};
				}
			}
		}

		private class HeroPairing {
			public HeroType MyHero;
			public HeroType EnemyHero;
			public double WinRate;
		}
	}
}
