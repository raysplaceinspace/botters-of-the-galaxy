using System;
using System.Collections.Generic;

namespace BOTG_Refree
{
	public enum SkillType
	{
		SELF, UNIT, POSITION
	}

	//Skill is something the hero uses
	public abstract class SkillBase
	{
		public readonly Hero hero;
		public readonly int manaCost;
		public readonly string skillName;
		public readonly int range;
		public int cooldown;
		public int initialCooldown;
		public double duration = 1;

		protected SkillBase(Hero hero, int manaCost, string skillName, int range, int cooldown)
		{
			this.hero = hero;
			this.manaCost = manaCost;
			this.skillName = skillName;
			this.range = range;
			this.initialCooldown = cooldown;
			this.cooldown = 0;
		}

		public abstract double CastTime();
		internal abstract void doSkill(Game game, double x, double y, int unitId);

		public int getDuration() { return (duration < 1 ? 1 : (int)(Math.Round(duration))); }
		public abstract string getTargetTeam();
		public abstract SkillType getTargetType();
	}

	public class Skills
	{
		public class EmptySkill : SkillBase
		{
			public EmptySkill() : base(null, 100000, "NONE", 0, Const.Rounds + 1) { }

			override internal void doSkill(Game game, double x, double y, int unitId) { }

			override public string getTargetTeam()
			{
				return "NONE";
			}

			override public SkillType getTargetType() { return SkillType.SELF; }
			override public double CastTime() { return 0.0; }
		}

		public class BlinkSkill : SkillBase
		{
			bool instant;
			public BlinkSkill(Hero hero, int manaCost, string skillName, double duration, bool instant, int range, int cooldown) : base(hero, manaCost, skillName, range, cooldown)
			{
				this.duration = duration;
				this.instant = instant;
			}

			override internal void doSkill(Game game, double x, double y, int unitId)
			{
				Point target = new Point(x, y);
				if (target.Distance(hero) > range)
				{
					double distance = target.Distance(hero);
					target.x = hero.x + ((target.x - hero.x) / distance * range);
					target.y = hero.y + ((target.y - hero.y) / distance * range);
				}

				hero.mana = Math.Min(hero.mana + 20, hero.maxMana);

				game.events.Add(new BlinkEvent(hero, duration, Utilities.round(target.x), Utilities.round(target.y)));
			}

			override public double CastTime() { return duration; }

			override public string getTargetTeam()
			{
				return "NONE";
			}

			override public SkillType getTargetType()
			{
				return SkillType.POSITION;
			}
		}

		// Lancer skills
		public class JumpSkill : SkillBase
		{
			public JumpSkill(Hero hero, int manaCost, string skillName, double duration, int range, int cooldown) :
				base(hero, manaCost, skillName, range, cooldown)
			{
				this.duration = duration;
			}

			override internal void doSkill(Game game, double x, double y, int unitId)
			{
				Point target = new Point(x, y);
				double distance = target.Distance(hero);
				if (distance > range)
				{
					target.x = hero.x + ((target.x - hero.x) / distance * range);
					target.y = hero.y + ((target.y - hero.y) / distance * range);
				}

				game.events.Add(new BlinkEvent(hero, duration, target.x, target.y));
				game.events.Add(new AttackNearestDelayed(hero, duration + Const.EPSILON));
			}

			override public double CastTime() { return duration; }

			override public string getTargetTeam()
			{
				return "ENEMY";
			}

			override public SkillType getTargetType()
			{
				return SkillType.POSITION;
			}
		}

		public class FlipSkill : SkillBase
		{
			public FlipSkill(Hero hero, int manaCost, string skillName, double duration, int range, int cooldown) :
			base(hero, manaCost, skillName, range, cooldown)
			{
				this.duration = duration;
			}

			override internal void doSkill(Game game, double x, double y, int unitId)
			{
				Unit target = Const.game.getUnitOfId(unitId);
				if (target.Distance(hero) <= range && !(target is Tower))
				{
					game.events.Add(new StunEvent(target, 0, 1));
					if (target.team != hero.team)
						game.events.Add(new DamageEvent(target, hero, duration, (int)(hero.damage * 0.4)));

					double vx = (hero.x - target.x) * 2 / duration;
					double vy = (hero.y - target.y) * 2 / duration;
					game.events.Add(new SpeedChangedForceEvent(target, Const.EPSILON, vx, vy));
					game.events.Add(new SpeedChangedForceEvent(target, duration, vx * -1, vy * -1));
				} 
			}
			override public double CastTime() { return 0.0; }

			override public string getTargetTeam()
			{
				return "BOTH";
			}

			override public SkillType getTargetType()
			{
				return SkillType.UNIT;
			}
		}

		public class PowerUpSkill : SkillBase
		{
			public PowerUpSkill(Hero hero, int manaCost, string skillName, double duration, int range, int cooldown) :
				base(hero, manaCost, skillName, range, cooldown)
			{
				this.duration = duration;
			}

			override internal void doSkill(Game game, double x, double y, int unitId)
			{
				int dmgIncrease = (int)(hero.moveSpeed * Const.POWERUPDAMAGEINCREASE);
				hero.moveSpeed += 0;
				hero.range += 10;
				hero.damage += dmgIncrease;
				game.events.Add(new PowerUpEvent(hero, 0, 10, dmgIncrease, (int)duration));
			}
			override public double CastTime() { return 0.0; }

			override public string getTargetTeam()
			{
				return "NONE";
			}

			override public SkillType getTargetType()
			{
				return SkillType.SELF;
			}
		}
		// Lancer skills end

		// AOE health Skills
		public class AOEHealSkill : SkillBase
		{
			private string skin;
			int radius;

			public AOEHealSkill(Hero hero, int manaCost, string skillName, double duration, int range, int radius, string skin, int cooldown) :
			base(hero, manaCost, skillName, range, cooldown)
			{
				this.duration = duration;
				this.radius = radius;
				this.skin = skin;
			}

			override internal void doSkill(Game game, double x, double y, int unitId)
			{
				Point target = new Point(x, y);
				if (target.Distance(hero) <= range)
				{
					game.events.Add(new HealthChangeEvent(target, duration, radius, (int)(0.2 * (hero.mana + manaCost)), false, hero));
				}
			}
			override public double CastTime() { return duration; }

			override public string getTargetTeam()
			{
				return "ALLIED";
			}

			override public SkillType getTargetType()
			{
				return SkillType.POSITION;
			}
		}


		public class BurningGround : SkillBase
		{
			private string skin;
			int radius;

			public BurningGround(Hero hero, int manaCost, string skillName, double duration, int range, int radius, string skin, int cooldown) :
			base(hero, manaCost, skillName, range, cooldown)
			{
				this.duration = duration;
				this.radius = radius;
				this.skin = skin;
			}

			override internal void doSkill(Game game, double x, double y, int unitId)
			{
				Point target = new Point(x, y);
				if (target.Distance(hero) <= range + Const.EPSILON)
				{
					game.events.Add(new HealthChangeEvent(target, duration, radius, -1 * (hero.manaregeneration * 5 + 30), true, hero));
				}
			}
			override public double CastTime() { return duration; }

			override public string getTargetTeam()
			{
				return "ENEMY";
			}

			override public SkillType getTargetType()
			{
				return SkillType.POSITION;
			}
		}


		public class ShieldSkill : SkillBase
		{
			public ShieldSkill(Hero hero, int manaCost, string skillName, int duration, int range, int cooldown) :
				base(hero, manaCost, skillName, range, cooldown)
			{
				this.duration = duration;
			}


            override internal void doSkill(Game game, double x, double y, int unitId)
			{
				Unit target = Const.game.getUnitOfId(unitId);
				if (target.Distance(hero) <= range)
				{
					target.shield = Math.Max(target.shield, (int)(0.5 * hero.maxMana + 50));
					game.events.Add(new ShieldEvent(target, (int)duration));
				} 
			}

			override public double CastTime() { return 0.0; }

			override public string getTargetTeam()
			{
				return "ALLIED";
			}

			override public SkillType getTargetType()
			{
				return SkillType.UNIT;
			}
		}

		public class PullSkill : SkillBase
		{
			double delay;
			public PullSkill(Hero hero, int manaCost, string skillName, double duration, int range, int cooldown, double delay) :
				base(hero, manaCost, skillName, range, cooldown)
			{
				this.duration = duration;
				this.delay = delay;
			}

			override internal void doSkill(Game game, double x, double y, int unitId)
			{
				Unit unit = Const.game.getUnitOfId(unitId);
				double distance = unit.Distance(hero);
				if (distance <= range && !(unit is Tower))
				{
					if (distance > Const.EPSILON)
					{
						double vx = (hero.x - unit.x) / distance * 200 / duration;
						double vy = (hero.y - unit.y) / distance * 200 / duration;
						game.events.Add(new SpeedChangedForceEvent(unit, delay, vx, vy));
						game.events.Add(new SpeedChangedForceEvent(unit, delay + duration, vx * -1, vy * -1));
						game.events.Add(new StunEvent(unit, delay, 1));
					}

					if (unit is Hero && unit.team != hero.team)
						game.events.Add(new DrainManaEvent(unit, delay + duration, ((Hero)unit).manaregeneration * 3 + 5, hero));

				}
			}
			override public double CastTime() { return delay; }

			override public string getTargetTeam()
			{
				return "BOTH";
			}

			override public SkillType getTargetType()
			{
				return SkillType.UNIT;
			}
		}
		// DR STRANGE skills end

		// HULK Skills
		public class ChargeSkill : SkillBase
		{
			public ChargeSkill(Hero hero, int manaCost, string skillName, double duration, int range, int cooldown) :
				base(hero, manaCost, skillName, range, cooldown)
			{
				this.duration = duration;
			}

			override internal void doSkill(Game game, double x, double y, int unitId)
			{
				Unit target = Const.game.getUnitOfId(unitId);
				double distance = target.Distance(hero);
				if (distance <= range)
				{

					game.events.Add(new BlinkEvent(hero, duration, target.x, target.y));
					if (target.team != hero.team)
					{

						//Reduce dmg on his attack on the delayed attack
						int halfDmg = hero.damage / 2;
						hero.damage -= halfDmg;
						game.events.Add(new PowerUpEvent(hero, 0, 0, -halfDmg, 0));
						game.events.Add(new DelayedAttackEvent(target, hero, duration + Const.EPSILON));

						game.events.Add(new PowerUpEvent(target, 150, 0, 0, 0));
						game.events.Add(new PowerUpEvent(target, -150, 0, 0, 3));
					}
				} 
			}
			override public double CastTime() { return duration; }

			override public string getTargetTeam()
			{
				return "ENEMY";
			}

			override public SkillType getTargetType()
			{
				return SkillType.UNIT;
			}
		}

		public class BashSkill : SkillBase
		{
			public BashSkill(Hero hero, int manaCost, string skillName, int duration, int range, int cooldown) :
				base(hero, manaCost, skillName, range, cooldown)
			{
				this.duration = duration;
			}

			override internal void doSkill(Game game, double x, double y, int unitId)
			{
				Unit target = Const.game.getUnitOfId(unitId);
				if (target.Distance(hero) <= range && !(target is Tower))
				{
					game.events.Add(new DamageEvent(target, hero, hero.attackTime, hero.damage));
					game.events.Add(new StunEvent(target, hero.attackTime, (int)duration));
				} 
			}
			override public double CastTime() { return hero.attackTime; }

			override public string getTargetTeam()
			{
				return "ENEMY";
			}

			override public SkillType getTargetType()
			{
				return SkillType.UNIT;
			}
		}

		public class ExplosiveSkill : SkillBase
		{
			public ExplosiveSkill(Hero hero, int manaCost, string skillName, double duration, int range, int cooldown) :
				base(hero, manaCost, skillName, range, cooldown)
			{
				this.duration = duration;
			}

			override internal void doSkill(Game game, double x, double y, int unitId)
			{
				hero.explosiveShield = (int)(hero.maxHealth * 0.07 + 50);
				game.events.Add(new ExplosiveShieldEvent(hero, (int)Math.Round(duration)));
			}
			override public double CastTime() { return 0.0; }

			override public string getTargetTeam()
			{
				return "ENEMY";
			}

			override public SkillType getTargetType()
			{
				return SkillType.SELF;
			}
		}
		// Knight Skills end

		// Ninja Skills
		public class CounterSkill : SkillBase
		{
			public CounterSkill(Hero hero, int manaCost, string skillName, int duration, int range, int cooldown) :
				base(hero, manaCost, skillName, range, cooldown)
			{
				this.duration = duration;
			}

			override internal void doSkill(Game game, double x, double y, int unitId)
			{
				game.events.Add(new CounterEvent(hero, duration - Const.EPSILON, range));
			}
			override public double CastTime() { return 0.0; }

			override public string getTargetTeam()
			{
				return "ENEMY";
			}

			override public SkillType getTargetType()
			{
				return SkillType.SELF;
			}
		}

		public class StealthSkill : SkillBase
		{
			public StealthSkill(Hero hero, int manaCost, string skillName, double range, double duration, int cooldown) :
				base(hero, manaCost, skillName, 0, cooldown)
			{
				this.duration = duration;
			}

			override internal void doSkill(Game game, double x, double y, int unitId)
			{
				hero.invisibleBySkill = true;
				game.events.Add(new StealthEvent(hero, duration, hero.mana));
				hero.runTowards(new Point(x, y), hero.moveSpeed);
			}
			override public double CastTime() { return 1.0; }

			override public string getTargetTeam()
			{
				return "NONE";
			}

			override public SkillType getTargetType()
			{
				return SkillType.POSITION;
			}
		}

		public class WireHookSkill : SkillBase
		{
			double radius;
			int stun_time;
			double speed;
			double flyTime;


			public WireHookSkill(Hero hero, int manaCost, string skillName, int range, int radius, int stun_time, double duration, int cooldown) :
				base(hero, manaCost, skillName, range, cooldown)
			{
				this.stun_time = stun_time;
				this.speed = range / duration;
				this.radius = radius;
				this.flyTime = duration;
				this.duration = stun_time;
			}

			override internal void doSkill(Game game, double x, double y, int unitId)
			{
				Point target = new Point(x, y);
				double distance = hero.Distance(target);
				target.x = hero.x + (x - hero.x) / distance * range;
				target.y = hero.y + (y - hero.y) / distance * range;

				MovingEntity lineSpellUnit = new MovingEntity(hero.x, hero.y, (target.x - hero.x) / flyTime, (target.y - hero.y) / flyTime);

				double lowestT = 2;
				List<Hero> possibleTargets = new List<Hero>();
				for (int i = Const.game.allUnits.Count - 1; i >= 0; i--)
				{
					Unit unit = Const.game.allUnits[i];
					if (unit.team != hero.team && (unit is Hero))
					{
						double collisionT = unit.getCollisionTime(lineSpellUnit, radius);
						possibleTargets.Add((Hero)unit);
						if (collisionT >= 0 && collisionT <= lowestT)
						{
							lowestT = collisionT;
						}
					}
				}

				Const.game.events.Add(new WireEvent(possibleTargets, lowestT, lineSpellUnit, stun_time, hero, radius, duration, 0.5));
			}
			override public double CastTime() { return 0.0; }

			override public string getTargetTeam()
			{
				return "ENEMY";
			}

			override public SkillType getTargetType()
			{
				return SkillType.POSITION;
			}
		}
		// Ninja Skills end

		public class LineSkill : SkillBase
		{
			double radius;
			double speed;


			public LineSkill(Hero hero, int manaCost, string skillName, int range, int radius, double duration, int cooldown) :
				base(hero, manaCost, skillName, range, cooldown)
			{
				this.speed = range / duration;
				this.radius = radius;
				this.duration = duration;
			}

			override internal void doSkill(Game game, double x, double y, int unitId)
			{
				Point target = new Point(x, y);
				double distance = hero.Distance(target);
				double vx = (x - hero.x) / distance;
				double vy = (y - hero.y) / distance;

				target.x = hero.x + vx * range;
				target.y = hero.y + vy * range;

				MovingEntity lineSpellUnit = new MovingEntity(hero.x, hero.y, vx * range / duration, vy * range / duration);

				foreach (Unit unit in Const.game.allUnits)
				{
					if (unit.team != hero.team && (unit is Hero || unit is Creature))
					{
						double collisionT = unit.getCollisionTime(lineSpellUnit, radius - Const.EPSILON);
						Const.game.events.Add(new LineEffectEvent(unit, collisionT < 0 ? duration : collisionT, lineSpellUnit, (int)(0.2 * (hero.mana + manaCost)), hero, radius, duration, 55));
					}
				}
			}
			override public double CastTime() { return 0.0; }

			override public string getTargetTeam()
			{
				return "ENEMY";
			}

			override public SkillType getTargetType()
			{
				return SkillType.POSITION;
			}
		}
	}
}