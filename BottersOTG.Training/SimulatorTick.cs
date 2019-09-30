using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BottersOTG.Model;
using Utils;

namespace BottersOTG.Training {
	public class SimulatorTick {
		public const double Precision = 0.0001;

		public World World;
		public Dictionary<int, GameAction> Actions;

		private readonly Dictionary<int, Unit> _units;
		private readonly Dictionary<int, double> _timeSpent = new Dictionary<int, double>();

		private readonly Dictionary<int, List<int>> _attackersPerUnit = new Dictionary<int, List<int>>();
		private readonly Dictionary<int, double> _damageOutputPerUnit = new Dictionary<int, double>();
		
		private SimulatorTick(World world, Dictionary<int, GameAction> actions) {
			World = world;
			Actions = actions;
			_units = world.Units.ToDictionary(u => u.UnitId);
		}

		public static World Forward(World world, Dictionary<int, GameAction> actions) {
			World nextWorld = world.Clone();
			++nextWorld.Tick;

			new SimulatorTick(nextWorld, actions).Forward();

			return nextWorld;
		}

		private void Forward() {
			// Timers/mana regen
			foreach (Unit unit in _units.Values) {
				if (unit.AggroTicksRemaining > 0) {
					--unit.AggroTicksRemaining;
				}
				if (unit.CountDown1 > 0) {
					--unit.CountDown1;
				}
				if (unit.CountDown2 > 0) {
					--unit.CountDown2;
				}
				if (unit.CountDown3 > 0) {
					--unit.CountDown3;
				}
				if (unit.StunDuration > 0) {
					--unit.StunDuration;
				}
				if (unit.ShieldTicksRemaining > 0) {
					--unit.ShieldTicksRemaining;
					if (unit.ShieldTicksRemaining == 0) {
						unit.Shield = 0;
					}
				}

				if (unit.StealthTicksRemaining > 0) {
					--unit.StealthTicksRemaining;
				}
				unit.IsVisible = unit.StealthTicksRemaining == 0;

				if (unit.ChargeTicksRemaining > 0) {
					--unit.ChargeTicksRemaining;
				}

				if (unit.ExplosiveShieldCooldown > 0) {
					--unit.ExplosiveShieldCooldown;
					if (unit.ExplosiveShieldCooldown == 0) {
						unit.Shield = 0;
					}
				}

				if (unit.PowerupTicksRemaining > 0) {
					--unit.PowerupTicksRemaining;
				}

				unit.Mana = Math.Min(unit.MaxMana, unit.Mana + unit.ManaRegeneration);
			}

			// Spells
			List<CounterState> reflectors = new List<CounterState>();
			foreach (KeyValuePair<int, GameAction> kvp in Actions.OrderBy(kvp => kvp.Value.ActionType)) {
				Unit unit = _units.GetOrDefault(kvp.Key);
				GameAction action = kvp.Value;

				if (unit == null || action == null || unit.StunDuration > 0) {
					continue;
				}

				if (action.ActionType == ActionType.Blink) {
					Blink(unit, action.Target);
				} else if (action.ActionType == ActionType.AoeHeal) {
					AoeHeal(unit, action.Target, World.Units);
				} else if (action.ActionType == ActionType.Shield) {
					Shield(unit, _units.GetOrDefault(action.UnitId));
				} else if (action.ActionType == ActionType.Fireball) {
					Fireball(unit, action.Target, World.Units);
				} else if (action.ActionType == ActionType.Burning) {
					Burning(unit, action.Target, World.Units);
				} else if (action.ActionType == ActionType.Pull) {
					Pull(unit, _units.GetOrDefault(action.UnitId));
				} else if (action.ActionType == ActionType.Counter) {
					CounterState state = CounterStart(unit);
					if (state != null) {
						reflectors.Add(state);
					}
				} else if (action.ActionType == ActionType.Wire) {
					Wire(unit, action.Target, World.Units);
				} else if (action.ActionType == ActionType.Stealth) {
					Stealth(unit);
				} else if (action.ActionType == ActionType.Charge) {
					Charge(unit, _units.GetOrDefault(action.UnitId));
				} else if (action.ActionType == ActionType.ExplosiveShield) {
					ExplosiveShield(unit);
				} else if (action.ActionType == ActionType.Bash) {
					Bash(unit, _units.GetOrDefault(action.UnitId));
				} else if (action.ActionType == ActionType.SpearFlip) {
					SpearFlip(unit, _units.GetOrDefault(action.UnitId));
				} else if (action.ActionType == ActionType.Jump) {
					Jump(unit, action.Target, World.Units);
				} else if (action.ActionType == ActionType.Powerup) {
					Powerup(unit);
				}
			}

			// Move
			foreach (Unit unit in _units.Values) {
				if (unit.MovementSpeed <= 0 || unit.StunDuration > 0) {
					continue;
				}

				GameAction action = Actions.GetOrDefault(unit.UnitId);
				if (action == null) {
					continue;
				}

				Vector? target = null;
				if (action.ActionType == ActionType.Move || action.ActionType == ActionType.AttackMove) {
					target = action.Target;
				} else if (action.ActionType == ActionType.AttackNearest || action.ActionType == ActionType.Attack) {
					Unit targetUnit;
					if (action.ActionType == ActionType.AttackNearest) {
						int enemyTeam = World.Enemy(unit.Team);
						targetUnit = _units.Values.Where(u => u.Team == enemyTeam && u.IsVisible).MinByOrDefault(x => unit.Pos.DistanceTo(x.Pos));
					} else if (action.ActionType == ActionType.Attack) {
						targetUnit = _units.GetOrDefault(action.UnitId);
					} else {
						targetUnit = null;
					}
					if (targetUnit != null) {
						if (unit.Pos.DistanceTo(targetUnit.Pos) <= unit.AttackRange) {
							target = null;
						} else {
							Vector offset = targetUnit.Pos.Minus(unit.Pos);
							double moveDistance = offset.Length() - unit.AttackRange;
							target = unit.Pos.Plus(offset.Unit().Multiply(moveDistance));
						}
					}
				}

				if (target != null) {
					int movementSpeed = unit.MovementSpeed;
					if (unit.ChargeTicksRemaining > 0) {
						movementSpeed -= World.ChargeMovePenalty;
					}

					Vector step = target.Value.Minus(unit.Pos);
					if (step.Length() > movementSpeed) {
						step = step.Unit().Multiply(movementSpeed);
					}
					unit.Pos = unit.Pos.Plus(step);

					double timeUsed = step.Length() / movementSpeed;
					_timeSpent[unit.UnitId] = _timeSpent.GetOrDefault(unit.UnitId) + timeUsed;
				}
			}

			// Attack
			foreach (Unit unit in World.Units) {
				if (unit.StunDuration > 0) {
					continue;
				}

				GameAction action = Actions.GetOrDefault(unit.UnitId);
				if (action == null) {
					continue;
				}

				Unit target = null;
				if (action.ActionType == ActionType.Attack ||
					action.ActionType == ActionType.AttackMove ||
					action.ActionType == ActionType.Bash ||
					action.ActionType == ActionType.Charge) {
					target = _units.GetOrDefault(action.UnitId);
				} else if (action.ActionType == ActionType.AttackNearest) {
					int enemyTeam = World.Enemy(unit.Team);
					target = _units.Values.Where(u => u.Team == enemyTeam && u.IsVisible).MinByOrDefault(x => unit.Pos.DistanceTo(x.Pos));
				}

				int attackRange = unit.AttackRange;
				double damage = unit.AttackDamage;
				if (unit.PowerupTicksRemaining > 0) {
					attackRange += World.PowerupBonusAttackRange;
					damage += World.PowerupAttackBonus(unit.MovementSpeed);
				}

				if (target != null && (target.IsVisible || unit.IsDetector)) {
					double distance = unit.Pos.DistanceTo(target.Pos);
					double attackTime = World.AttackTime(distance, attackRange, unit.UnitType);
					double newTimeSpent = _timeSpent.GetOrDefault(unit.UnitId) + attackTime;
					if (distance <= unit.AttackRange + Precision && newTimeSpent <= 1.0 + Precision) {
						Damage(target, damage, unit.UnitId);
						_timeSpent[unit.UnitId] = newTimeSpent;
					}
				}
			}

			// Explosive shield
			foreach (Unit unit in World.Units) {
				if (unit.ExplosiveShieldTicksRemaining > 0 && unit.Shield == 0) {
					unit.ExplosiveShieldTicksRemaining = 0;
					ExplodeShield(unit, World.Units);
				}
			}

			// Counter
			foreach (CounterState state in reflectors) {
				Unit deadpool = _units.GetOrDefault(state.UnitId);
				if (deadpool != null && deadpool.Health > 0) {
					CounterFinish(deadpool, state, World.Units);
				}
			}

			// Aggro
			{
				HashSet<Unit> aggravators = new HashSet<Unit>();
				foreach (Unit unit in World.Units) {
					if (unit.UnitType != UnitType.Hero) {
						continue;
					}

					IEnumerable<Unit> heroicAttackers =
						AttackersOf(unit.UnitId)
						.WhereNotNull()
						.Where(u => u.UnitType == UnitType.Hero);
					foreach (Unit heroicAttacker in heroicAttackers) {
						aggravators.Add(heroicAttacker);
					}
				}

				foreach (Unit aggravator in aggravators) {
					foreach (Unit enemyUnit in World.Units.Where(u => u.Team == World.Enemy(aggravator.Team))) {
						if (enemyUnit.UnitType == UnitType.Minion && enemyUnit.Pos.DistanceTo(aggravator.Pos) <= World.AggroRange) {
							enemyUnit.AggroTargetUnitId = aggravator.UnitId;
							enemyUnit.AggroTicksRemaining = 3;
						} else if (enemyUnit.UnitType == UnitType.Tower && enemyUnit.Pos.DistanceTo(aggravator.Pos) <= enemyUnit.AttackRange) {
							enemyUnit.AggroTargetUnitId = aggravator.UnitId;
							enemyUnit.AggroTicksRemaining = 3;
						}
					}
				}
			}

			// Death
			{
				List<Unit> dead = World.Units.Where(u => u.Health <= 0).ToList();
				World.Units = World.Units.Except(dead).ToList();

				foreach (Unit unit in dead) {
					int[] lastHitTeams = AttackersOf(unit.UnitId).Select(attacker => attacker.Team).Distinct().ToArray();
					bool denied = lastHitTeams.Any(team => team == unit.Team);
					if (denied) {
						++World.Denies[unit.Team];
						if (unit.UnitType == UnitType.Hero) {
							World.Gold[unit.Team] += 150; // Denied hero bonus
						}
					} else {
						foreach (int lastHitTeam in lastHitTeams) {
							if (lastHitTeam == -1) {
								continue;
							}

							++World.MinionsKilled[lastHitTeam];
							World.Gold[lastHitTeam] += unit.GoldValue;
						}
					}
				}
			}

			// Update damage output totals
			foreach (KeyValuePair<int, double> kvp in _damageOutputPerUnit) {
				double damageOutput = kvp.Value;
				Unit unit = _units[kvp.Key];

				if (unit.Team != -1 && unit.UnitType == UnitType.Hero) {
					World.HeroDamageOutput[unit.Team] += damageOutput;
				}
			}

			// New minions
			if (World.SpawnMinions && World.Tick % World.SpawnMinionsInterval == 0) {
				WorldGenerator.AddMinions(World);
			}
		}

		private void Damage(Unit target, double damage, int attackerId) {
			target.Damage(damage);

			_damageOutputPerUnit[attackerId] = _damageOutputPerUnit.GetOrDefault(attackerId) + damage;

			List<int> attackerIds;
			if (!_attackersPerUnit.TryGetValue(target.UnitId, out attackerIds)) {
				attackerIds = _attackersPerUnit[target.UnitId] = new List<int>();
			}
			attackerIds.Add(attackerId);
		}

		private IEnumerable<Unit> AttackersOf(int targetUnitId) {
			List<int> attackerIds = _attackersPerUnit.GetOrDefault(targetUnitId);
			if (attackerIds == null) {
				yield break;
			}

			foreach (int attackerId in attackerIds) {
				yield return _units[attackerId];
			}
		}

		private void Blink(Unit hero, Vector target) {
			if (!(hero.UnitType == UnitType.Hero &&
				hero.HeroType == HeroType.Ironman &&
				hero.BlinkCooldown == 0 &&
				hero.Mana >= World.BlinkCost &&
				hero.Pos.DistanceTo(target) <= World.BlinkRange)) {
				return;
			}

			hero.Pos = target;
			hero.Mana -= World.BlinkCost;
			hero.Mana += World.BlinkReplenish;
			hero.BlinkCooldown = World.BlinkCooldown;
		}

		private void Fireball(Unit hero, Vector target, IEnumerable<Unit> units) {
			if (!(hero.UnitType == UnitType.Hero &&
				hero.HeroType == HeroType.Ironman &&
				hero.FireballCooldown == 0 &&
				hero.Mana >= World.FireballCost)) {
				return;
			}

			Vector shootDirection = target.Minus(hero.Pos).Unit();
			foreach (Unit unit in units) {
				if (unit.Team != World.Enemy(hero.Team)) {
					continue;
				}

				Vector? collision = Geometry.LinearCollision(hero.Pos, shootDirection, unit.Pos, World.FireballRadius);
				if (!collision.HasValue) {
					continue;
				}

				double fireballDistance = hero.Pos.DistanceTo(collision.Value);
				if (fireballDistance > World.FireballRange) {
					continue;
				}

				double fireballDamage = World.FireballDamage(hero.Mana, fireballDistance);
				Damage(unit, fireballDamage, hero.UnitId);
			}

			hero.Mana -= World.FireballCost;
			hero.FireballCooldown = World.FireballCooldown;
		}

		private void Burning(Unit hero, Vector target, List<Unit> units) {
			if (!(hero.UnitType == UnitType.Hero &&
				hero.HeroType == HeroType.Ironman &&
				hero.BurningCooldown == 0 &&
				hero.Mana >= World.FireballCost &&
				hero.Pos.DistanceTo(target) <= World.BurningRange)) {
				return;
			}

			double burningDamage = World.BurningDamage(hero.ManaRegeneration);
			foreach (Unit unit in units) {
				if (unit.Team == World.Enemy(hero.Team) &&
					unit.Pos.DistanceTo(target) <= World.BurningRadius) {
					Damage(unit, burningDamage, hero.UnitId);
				}
			}

			hero.Mana -= World.BurningCost;
			hero.BurningCooldown = World.BurningCooldown;
		}

		private void AoeHeal(Unit hero, Vector target, IEnumerable<Unit> units) {
			if (!(hero.UnitType == UnitType.Hero &&
				hero.HeroType == HeroType.DoctorStrange &&
				hero.AoeHealCooldown == 0 &&
				hero.Mana >= World.AoeHealCost &&
				hero.Pos.DistanceTo(target) <= World.AoeHealRange)) {
				return;
			}

			double healing = World.AoeHealing(hero.Mana);
			foreach (Unit unit in units) {
				if (unit.Team == hero.Team &&
					unit.Pos.DistanceTo(target) <= World.AoeHealRadius) {

					unit.Heal(healing);
				}
			}

			hero.Mana -= World.AoeHealCost;
			hero.AoeHealCooldown = World.AoeHealCooldown;
		}

		private void Shield(Unit hero, Unit target) {
			if (target == null) {
				return;
			}

			if (!(hero.UnitType == UnitType.Hero &&
				hero.HeroType == HeroType.DoctorStrange &&
				hero.ShieldCooldown == 0 &&
				hero.Mana >= World.ShieldCost &&
				target.IsVisible &&
				hero.Pos.DistanceTo(target.Pos) <= World.ShieldRange)) {
				return;
			}

			target.Shield += (int)Math.Round(World.ShieldBonus(hero.MaxMana));
			target.ShieldTicksRemaining = World.ShieldDuration;

			hero.Mana -= World.ShieldCost;
			hero.ShieldCooldown = World.ShieldCooldown;
		}

		private void Pull(Unit hero, Unit target) {
			if (target == null) {
				return;
			}

			if (!(hero.UnitType == UnitType.Hero &&
				hero.HeroType == HeroType.DoctorStrange &&
				hero.PullCooldown == 0 &&
				hero.Mana >= World.PullCost &&
				target.IsVisible &&
				hero.Pos.DistanceTo(target.Pos) <= World.PullRange)) {
				return;
			}

			if (hero.Pos.DistanceTo(target.Pos) <= World.PullDistance) {
				target.Pos = hero.Pos;
			} else {
				Vector pullDirection = hero.Pos.Minus(target.Pos).Unit();
				target.Pos = target.Pos.Plus(pullDirection.Multiply(World.PullDistance));
			}

			if (target.Team != hero.Team) {
				target.DrainMana(World.PullDrain(target.ManaRegeneration));
				target.Stun(World.PullStunDuration);
			}

			hero.Mana -= World.PullCost;
			hero.PullCooldown = World.PullCooldown;
		}

		private CounterState CounterStart(Unit hero) {
			if (!(hero.UnitType == UnitType.Hero &&
				hero.HeroType == HeroType.Deadpool &&
				hero.CounterCooldown == 0 &&
				hero.Mana >= World.CounterCost)) {
				return null;
			}

			hero.Mana -= World.CounterCost;
			hero.CounterCooldown = World.CounterCooldown;

			return new CounterState {
				UnitId = hero.UnitId,	
				InitialHealth = hero.Health,
			};
		}

		private void CounterFinish(Unit hero, CounterState state, IEnumerable<Unit> units) {
			if (hero.Health <= 0) {
				return;
			}

			Unit nearestEnemy =
				units.Where(u => u.Team == World.Enemy(hero.Team))
				.MinByOrDefault(u => u.Pos.DistanceTo(hero.Pos));
			if (nearestEnemy == null || nearestEnemy.Pos.DistanceTo(hero.Pos) > World.CounterRange) {
				return;
			}

			double counterDamage = World.CounterDamage(state.InitialHealth - hero.Health);
			if (counterDamage <= 0) {
				return;
			}

			Damage(nearestEnemy, counterDamage, hero.UnitId);
			hero.Health = state.InitialHealth;
		}

		private void Wire(Unit hero, Vector target, IEnumerable<Unit> units) {
			if (!(hero.UnitType == UnitType.Hero &&
				hero.HeroType == HeroType.Deadpool &&
				hero.WireCooldown == 0 &&
				hero.Mana >= World.WireCost)) {
				return;
			}

			Vector shootDirection = target.Minus(hero.Pos).Unit();
			Unit closestHit = null;
			double closestDistance = double.MaxValue;
			foreach (Unit unit in units) {
				if (unit.Team != World.Enemy(hero.Team)) {
					continue;
				}

				Vector? collision = Geometry.LinearCollision(hero.Pos, shootDirection, unit.Pos, World.WireRadius);
				if (!collision.HasValue) {
					continue;
				}

				double distance = hero.Pos.DistanceTo(collision.Value);
				if (distance > World.WireRange) {
					continue;
				}

				if (distance < closestDistance) {
					closestDistance = distance;
					closestHit = unit;
				}
			}

			if (closestHit != null) {
				Damage(closestHit, World.WireDamage(closestHit.MaxMana), hero.UnitId);
				closestHit.Stun(World.WireStunDuration);
			}

			hero.Mana -= World.WireCost;
			hero.WireCooldown = World.WireCooldown;
		}

		private void Stealth(Unit hero) {
			if (!(hero.UnitType == UnitType.Hero &&
				hero.HeroType == HeroType.Deadpool &&
				hero.StealthCooldown == 0 &&
				hero.Mana >= World.StealthCost)) {
				return;
			}
			hero.Mana -= World.StealthCost;
			hero.StealthCooldown = World.StealthCooldown;
			hero.StealthTicksRemaining = World.StealthDuration + 1; // +1 because this tick is going to be visible and doesn't count
		}

		private void Charge(Unit hero, Unit target) {
			if (target == null) {
				return;
			}

			if (!(hero.UnitType == UnitType.Hero &&
				hero.HeroType == HeroType.Hulk &&
				hero.ChargeCooldown == 0 &&
				hero.Mana >= World.ChargeCost &&
				target.IsVisible &&
				hero.Pos.DistanceTo(target.Pos) <= World.ChargeRange)) {
				return;
			}

			hero.Mana -= World.ChargeCost;
			hero.ChargeCooldown = World.ChargeCooldown;

			hero.Pos = target.Pos;

			Damage(target, World.ChargeDamage(hero.AttackDamage), hero.UnitId);
			target.ChargeTicksRemaining = World.ChargeDuration;
		}

		private void ExplosiveShield(Unit hero) {
			if (!(hero.UnitType == UnitType.Hero &&
				hero.HeroType == HeroType.Hulk &&
				hero.ExplosiveShieldCooldown == 0 &&
				hero.Mana >= World.ExplosiveShieldCost)) {
				return;
			}
			hero.Mana -= World.ExplosiveShieldCost;
			hero.ExplosiveShieldCooldown = World.ExplosiveShieldCooldown;

			hero.Shield += (int)Math.Round(World.ExplosiveShieldBonus(hero.MaxHealth));
			hero.ExplosiveShieldTicksRemaining = World.ExplosiveShieldDuration;
		}

		private void ExplodeShield(Unit hero, IEnumerable<Unit> units) {
			hero.ExplosiveShieldTicksRemaining = 0;

			foreach (Unit unit in units) {
				if (unit.Team == World.Enemy(hero.Team) &&
					unit.Pos.DistanceTo(hero.Pos) <= World.ExplosiveShieldRadius) {

					Damage(unit, World.ExplosiveShieldDamage, hero.UnitId);
				}
			}
		}

		private void Bash(Unit hero, Unit target) {
			if (target == null) {
				return;
			}

			if (!(hero.UnitType == UnitType.Hero &&
				hero.HeroType == HeroType.Hulk &&
				hero.BashCooldown == 0 &&
				hero.Mana >= World.BashCost &&
				target.IsVisible &&
				hero.Pos.DistanceTo(target.Pos) <= World.BashRange)) {
				return;
			}

			hero.Mana -= World.BashCost;
			hero.BashCooldown = World.BashCooldown;

			hero.Pos = target.Pos;

			target.Stun(World.BashStunDuration);
		}

		private void SpearFlip(Unit hero, Unit target) {
			if (target == null) {
				return;
			}

			if (!(hero.UnitType == UnitType.Hero &&
				hero.HeroType == HeroType.Valkyrie &&
				hero.SpearFlipCooldown == 0 &&
				hero.Mana >= World.SpearFlipCost &&
				target.IsVisible &&
				hero.Pos.DistanceTo(target.Pos) <= World.SpearFlipRange)) {
				return;
			}

			hero.Mana -= World.SpearFlipCost;
			hero.SpearFlipCooldown = World.SpearFlipCooldown;

			Vector offset = target.Pos.Minus(hero.Pos);
			target.Pos = hero.Pos.Plus(offset.Reverse());

			if (target.Team != hero.Team) {
				target.Stun(World.SpearFlipStunDuration);
				Damage(target, World.SpearFlipDamage(hero.AttackDamage), hero.UnitId);
			}
		}

		private void Jump(Unit hero, Vector target, IEnumerable<Unit> units) {
			if (!(hero.UnitType == UnitType.Hero &&
				hero.HeroType == HeroType.Valkyrie &&
				hero.JumpCooldown == 0 &&
				hero.Mana >= World.JumpCost &&
				hero.Pos.DistanceTo(target) <= World.JumpRange)) {
				return;
			}

			hero.Pos = target;
			hero.JumpCooldown = World.JumpCooldown;
			hero.Mana -= World.JumpCost;

			Unit enemyToAttack =
				units
				.Where(u => u.Team == World.Enemy(hero.Team))
				.Select(u => new {
					Unit = u,
					Distance = u.Pos.DistanceTo(hero.Pos),
				})
				.Where(x => x.Distance <= hero.AttackRange)
				.MinByOrDefault(x => x.Distance)?.Unit;

			if (enemyToAttack != null) {
				Damage(enemyToAttack, hero.AttackDamage, hero.UnitId);
			}
		}

		private void Powerup(Unit hero) {
			if (!(hero.UnitType == UnitType.Hero &&
				hero.HeroType == HeroType.Valkyrie &&
				hero.PowerupCooldown == 0 &&
				hero.Mana >= World.PowerupCost)) {
				return;
			}

			hero.Mana -= World.PowerupCost;
			hero.PowerupCooldown = World.PowerupCooldown;
			hero.PowerupTicksRemaining = World.PowerupDuration;
		}

		private class CounterState {
			public int UnitId;
			public int InitialHealth;
		}
	}
}
