using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BottersOTG.Intelligence.Decisions;
using BottersOTG.Model;
using Utils;

namespace BottersOTG.Training {
	public static class Replayer {
		public const int TicksForward = 8;
		public const int NumLearningTicks = 4;
		public const double DiscountRate = 1.01;

		public static List<Episode> GenerateRollout(World world, int myHeroId, Policy myPolicy, Policy adverserialPolicy) {
			Unit myHero = world.Units.First(u => u.UnitId == myHeroId);
			int myTeam = myHero.Team;

			Policy[] policies = TeamPolicies(myTeam, myPolicy, adverserialPolicy);

			List<PartialRollout> rollouts =
				TacticsToEvaluate(myHero)
				.Select(tactic => GenerateRollout(world, policies, myTeam, myHeroId, tactic))
				.ToList();

			PartialRollout bestRollout = rollouts.MaxBy(x => x.Score);
			double weight = Math.Log(1 + bestRollout.Score - rollouts.Average(x => x.Score));
			if (weight > 0) {
				List<Episode> episodesToLearn = bestRollout.Episodes;
				foreach (Episode episode in episodesToLearn) {
					episode.Weight = weight;
				}
				return episodesToLearn;
			} else {
				return new List<Episode>();
			}
		}

		private static IEnumerable<Tactic> TacticsToEvaluate(Unit hero) {
			HeroType heroType = hero.HeroType;

			yield return Tactic.AttackSafely;
			yield return Tactic.AttackHero;
			yield return Tactic.Retreat;

#pragma warning disable 0162
			if (!World.EnableSpells) {
				yield break;
			}

			if (heroType == HeroType.Deadpool) {
				if (hero.StealthCooldown == 0 && hero.Mana >= World.StealthCost) {
					yield return Tactic.Stealth;
				}
				if (hero.WireCooldown == 0 && hero.Mana >= World.WireCost) {
					yield return Tactic.Wire;
				}
				if (hero.CounterCooldown == 0 && hero.Mana >= World.CounterCost) {
					yield return Tactic.Counter;
				}
			} else if (heroType == HeroType.DoctorStrange) {
				if (hero.AoeHealCooldown == 0 && hero.Mana >= World.AoeHealCost) {
					yield return Tactic.AoeHeal;
				}
				if (hero.ShieldCooldown == 0 && hero.Mana >= World.ShieldCost) {
					yield return Tactic.Shield;
				}
				if (hero.PullCooldown == 0 && hero.Mana >= World.PullCost) {
					yield return Tactic.Pull;
				}
			} else if (heroType == HeroType.Hulk) {
				if (hero.BashCooldown == 0 && hero.Mana >= World.BashCost) {
					yield return Tactic.Bash;
				}
				if (hero.ChargeCooldown == 0 && hero.Mana >= World.ChargeCost) {
					yield return Tactic.Charge;
				}
				if (hero.ExplosiveShieldCooldown == 0 && hero.Mana >= World.ExplosiveShieldCost) {
					yield return Tactic.ExplosiveShield;
				}
			} else if (heroType == HeroType.Ironman) {
				if (hero.FireballCooldown == 0 && hero.Mana >= World.FireballCost) {
					yield return Tactic.Fireball;
				}
				if (hero.BurningCooldown == 0 && hero.Mana >= World.BurningCost) {
					yield return Tactic.Burning;
				}
				if (hero.BlinkCooldown == 0 && hero.Mana >= World.BlinkCost) {
					yield return Tactic.Blink;
				}
			} else if (heroType == HeroType.Valkyrie) {
				if (hero.JumpCooldown == 0 && hero.Mana >= World.JumpCost) {
					yield return Tactic.Jump;
				}
				if (hero.SpearFlipCooldown == 0 && hero.Mana >= World.SpearFlipCost) {
					yield return Tactic.SpearFlip;
				}
				if (hero.PowerupCooldown == 0 && hero.Mana >= World.PowerupCost) {
					yield return Tactic.Powerup;
				}
			}
#pragma warning restore 0162
		}

		private static Policy[] TeamPolicies(int myTeam, Policy myPolicy, Policy adverserialPolicy) {
			return myTeam == 0 ? new[] { myPolicy, adverserialPolicy } : new[] { adverserialPolicy, myPolicy };
		}

		private static PartialRollout GenerateRollout(
			World current,
			Policy[] policies,
			int myTeam,
			int myHeroId,
			Tactic? initialTactic) {

			List<World> snapshots = new List<World> { current };
			List<Episode> episodes = new List<Episode>();
			for (int tick = 0; tick < NumLearningTicks + TicksForward; ++tick) {
				Dictionary<int, GameAction> actions = new Dictionary<int, GameAction>();
				Simulator.AddEnvironmentalActions(current, actions);

				foreach (Unit unit in current.Units) {
					if (unit.UnitType == UnitType.Hero) {
						Tactic tactic;
						if (tick == 0 && unit.UnitId == myHeroId && initialTactic.HasValue) {
							tactic = initialTactic.Value;
						} else {
							tactic = PolicyEvaluator.Evaluate(current, unit, policies[unit.Team]);
						}

						if (unit.UnitId == myHeroId && tick < NumLearningTicks) {
							episodes.Add(new Episode {
								World = current,
								Hero = unit,
								Tactic = tactic,
								Weight = 1.0,
							});
						}

						actions[unit.UnitId] = PolicyEvaluator.TacticToAction(current, unit, tactic);
					}
				}

				current = Simulator.Forward(current, actions);
				snapshots.Add(current);
			}

			List<IntermediateEvaluator.Evaluation> evaluations = snapshots.Select(world => IntermediateEvaluator.Evaluate(world, myTeam)).ToList();
			return new PartialRollout {
				Episodes = episodes,
				Evaluations = evaluations,
				Score = DiscountedRewards(evaluations.Select(ev => ev.Score), DiscountRate),
			};
		}

		private static double DiscountedRewards(IEnumerable<double> scores, double discountRate) {
			double total = 0.0;
			int count = 0;
			double? previousScore = null;
			foreach (double score in scores) {
				if (previousScore.HasValue) {
					double increment = score - previousScore.Value;
					total += increment / Math.Pow(discountRate, count++);
				}
				previousScore = score;
			}
			return total;
		}

		private class PartialRollout {
			public List<Episode> Episodes;
			public List<IntermediateEvaluator.Evaluation> Evaluations;
			public double Score;
		}

	}
}
