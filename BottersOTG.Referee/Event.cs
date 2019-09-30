using System;
using System.Collections.Generic;

namespace BOTG_Refree
{
	public abstract class Event: IComparable<Event>
	{
		public static int NONE = 0;
		public static int LIFECHANGED = 1;
		public static int SPEEDCHANGED = 2;
		public static int TELEPORTED = 4;
		public static int STUNNED = 8;

		protected static List<Unit> EMPTYLIST = new List<Unit>();

		protected double _t;
		internal double t;
		protected Unit unit;

		internal Event(Unit unit, double t)
		{
			this.unit = unit;
			this.t = t;
			_t = t;
		}

		public int CompareTo(Event compareEvent)
		{
			// A null value means that this object is greater.
			if (compareEvent == null)
				return 1;

			else
				return this.getOutcome().CompareTo(compareEvent.getOutcome());
		}
		internal virtual int getOutcome() { return NONE; }

		internal abstract List<Unit> onEventTime(double currentTime);

        internal virtual bool useAcrossRounds() { return false; }

		internal abstract bool afterAnotherEvent(Unit affectedUnit, int outcome, double currentTime);

		protected bool unitAlive(Unit unit, int outcome)
		{
			return !unitDead(this.unit, unit, outcome);
		}

		protected bool unitStopped(Unit unit)
		{
			if (unit.isDead) return true;
			if (unit.stunTime > 0) return true;
			return false;
		}

		protected bool unitStopped(Unit unit, Unit affected, int outcome)
		{
			if (unit != affected) return false;
			if (unitDead(unit, affected, outcome)) return true;
			if (hasOutcome(outcome, STUNNED) && unit.stunTime > 0) return true;
			return false;
		}

		protected void setSpeedAndAlertChange(Unit unit, double vx, double vy)
		{
			Const.game.events.Add(new SpeedChangedEvent(unit, 0.0, vx, vy));
		}

		protected bool unitDead(Unit unit, Unit affected, int outcome)
		{
			if (unit != affected) return false;
			return unit.isDead;
		}

		protected void runSilentlyTowards(Unit unit, Point targetPoint)
		{
			if (Math.Abs(unit.forceVY) > Const.EPSILON || Math.Abs(unit.forceVX) > Const.EPSILON) return;
			double targetDist = targetPoint.Distance(unit);
			double coef = (((double)unit.moveSpeed)) / targetDist;
			unit.vx = (targetPoint.x - unit.x) * coef;
			unit.vy = (targetPoint.y - unit.y) * coef;
			unit.moving = true;
		}

		protected bool hasAnyOutcome(int outcome, int expected1, int expected2)
		{
			return hasOutcome(outcome, expected1) || hasOutcome(outcome, expected2);
		}

		protected bool hasOutcome(int outcome, int expectedOutcome)
		{
			return (outcome & expectedOutcome) != 0;
		}


		protected Unit getClosestUnitInRange(Point root, double range, int team, bool allowInvis, Unit ignoredUnit)
		{
			double closestDist = double.MaxValue;
			Unit closest = null;
			foreach (Unit unit in Const.game.allUnits)
			{
				if (unit == ignoredUnit || unit.team == team) continue;
				if (unit is Tower) continue;
				if (!unit.visible && !allowInvis) continue;
				double dist = unit.Distance2(root);
				if (dist < closestDist && dist <= range * range)
				{
					closestDist = dist;
					closest = unit;
				}
			}

			return closest;
		}


        protected void doDamage(Unit unit, int damage, Unit attacker)
        {
            if (Const.game.damages.ContainsKey(unit))
                Const.game.damages[unit].Add(new Damage(unit, attacker, damage));
            else
            {
                List<Damage> damages = new List<Damage>();
                damages.Add(new Damage(unit, attacker, damage));
                Const.game.damages[unit] = damages;
            }
        }

		protected List<Unit> createListOfUnit()
		{
			List<Unit> units = new List<Unit>();
			units.Add(unit);
			return units;
		}
	}


	public class OnLocationReachedEvent : Event
	{
		public OnLocationReachedEvent(Unit unit, double t) : base(unit, t)
		{
		}

		internal override int getOutcome()
		{
			return SPEEDCHANGED;
		}

		override internal List<Unit> onEventTime(double currentTime)
		{
			unit.vx = unit.forceVX;
			unit.vy = unit.forceVY;
			unit.moving = false;

			return createListOfUnit();
		}

		override internal bool afterAnotherEvent(Unit affectedUnit, int outcome, double currentTime)
		{
			return unitStopped(unit, affectedUnit, outcome);
		}
	}

	public class SpeedChangedForceEvent : Event
	{
		double vx, vy;
		public SpeedChangedForceEvent(Unit unit, double t, double forcevx, double forcevy) :
			base(unit, t)
		{
			this.vx = forcevx;
			this.vy = forcevy;
		}

		override internal int getOutcome()
		{
			return SPEEDCHANGED;
		}

		override internal List<Unit> onEventTime(double currentTime)
		{
			unit.forceVX += vx;
			unit.forceVY += vy;
			unit.vx = unit.forceVX;
			unit.vy = unit.forceVY;

			return createListOfUnit();
		}

		override internal bool afterAnotherEvent(Unit affectedUnit, int outcome, double currentTime)
		{
			return !unitAlive(affectedUnit, outcome);
		}
	}

	public class SpeedChangedEvent : Event
	{
		double vx, vy;

		public SpeedChangedEvent(Unit unit, double t, double vx, double vy) : base(unit, t)
		{
			this.vx = vx;
			this.vy = vy;
		}

		override internal int getOutcome()
		{
			return SPEEDCHANGED;
		}

		override internal List<Unit> onEventTime(double currentTime)
		{
			if (unit.stunTime > 0 || (Math.Abs(unit.forceVY) > Const.EPSILON || Math.Abs(unit.forceVX) > Const.EPSILON)) return EMPTYLIST;
			unit.vx = vx;
			unit.vy = vy;

			return createListOfUnit();
		}

		override internal bool afterAnotherEvent(Unit affectedUnit, int outcome, double currentTime)
		{
			return !unitAlive(affectedUnit, outcome);
		}
	}


	public class DelayedAttackEvent : Event
	{
		Unit attacker;

		public DelayedAttackEvent(Unit unit, Unit attacker, double t) :
			 base(unit, t)
		{
			this.attacker = attacker;
		}

		override internal List<Unit> onEventTime(double currentTime)
		{
			this.attacker.fireAttack(this.unit);
			return EMPTYLIST;
		}

		override internal bool afterAnotherEvent(Unit affectedUnit, int outcome, double currentTime)
		{
			return unitStopped(this.attacker) || unitDead(this.unit, affectedUnit, outcome) || (affectedUnit == unit && hasOutcome(outcome, TELEPORTED));
		}
	}

	public class DamageEvent : Event
	{
		int damage;
		Unit attacker;

		public DamageEvent(Unit unit, Unit attacker, double t, int damage) :
			 base(unit, t)
		{
			this.attacker = attacker;
			this.damage = damage;
		}

		override internal List<Unit> onEventTime(double currentTime)
		{
			doDamage(unit, damage, attacker);
			return createListOfUnit();
		}

		override internal bool afterAnotherEvent(Unit affectedUnit, int outcome, double currentTime)
		{
			if (affectedUnit == this.unit && hasOutcome(outcome, TELEPORTED))
			{
				return true;
			}

			return unitDead(this.unit, affectedUnit, outcome);
		}
	}

	// when hit throws a dagger on nearby enemy
	public class CounterEvent : Event
	{
		int health, range;
		public CounterEvent(Unit defender, double t, int range) :
			base(defender, t)
		{
			this.health = defender.health;
			this.range = range;
		}

		override internal List<Unit> onEventTime(double currentTime)
		{
			if (this.health > unit.health)
			{
				Unit closest = getClosestUnitInRange(this.unit, this.range, this.unit.team, true, this.unit);
				int damage = (int)((this.health - unit.health) * 1.5);
				if (closest != null)
				{
					doDamage(closest, damage, this.unit);
				}

				unit.health = this.health;
			}
			return EMPTYLIST;
		}

		override internal bool afterAnotherEvent(Unit affectedUnit, int outcome, double currentTime)
		{
			return !unitAlive(affectedUnit, outcome);
		}
	}

	public class StunEvent : Event
	{
		int stunTime;
		public StunEvent(Unit unit, double t, int stunTime) :
			base(unit, t)
		{
			this.stunTime = stunTime;
		}

		override internal bool useAcrossRounds() { return true; }

		override internal int getOutcome()
		{
			return STUNNED;
		}

		override internal List<Unit> onEventTime(double currentTime)
		{
			unit.stunTime = Math.Max(unit.stunTime, stunTime);
			unit.vx = unit.forceVX;
			unit.vy = unit.forceVY;
			return createListOfUnit();
		}

		override internal bool afterAnotherEvent(Unit affectedUnit, int outcome, double currentTime)
		{
			return !unitAlive(affectedUnit, outcome);
		}
	}


	public class BlinkEvent : Event
	{
		double x, y;
		public BlinkEvent(Unit unit, double t, double x, double y) :
			 base(unit, t)
		{
			this.x = x;
			this.y = y;
		}

		override internal int getOutcome()
		{
			return TELEPORTED;
		}

		override internal List<Unit> onEventTime(double currentTime)
		{
			unit.move(x, y);
			return createListOfUnit();
		}

		override internal bool afterAnotherEvent(Unit affectedUnit, int outcome, double currentTime)
		{
			return unitDead(this.unit, affectedUnit, outcome);
		}
	}

	public class AttackNearestDelayed : Event
	{
		public AttackNearestDelayed(Hero hero, double t) :
			base(hero, t)
		{
		}

		override internal int getOutcome()
		{
			return NONE;
		}

		override internal List<Unit> onEventTime(double currentTime)
		{
			Unit toHit = getClosestUnitInRange(this.unit, this.unit.range, this.unit.team, false, this.unit);
			if (toHit != null) this.unit.fireAttack(toHit);
			return EMPTYLIST;
		}

		override internal bool afterAnotherEvent(Unit affectedUnit, int outcome, double currentTime)
		{
			return unitStopped(this.unit, affectedUnit, outcome);
		}
	}

	public class PowerUpEvent : Event
	{
		int moveSpeed, range, damage, rounds;
		public PowerUpEvent(Unit unit, int moveSpeed, int range, int damage, int rounds) :
			  base(unit, Const.MAXINT)
		{
			this.moveSpeed = moveSpeed;
			this.range = range;
			this.damage = damage;
			this.rounds = rounds;
		}

		override internal bool useAcrossRounds()
		{
			rounds--;
			if (rounds <= 0)
			{
				onEventTime(0);
				return false;
			}
			return true;
		}

		override internal int getOutcome()
		{
			return NONE;
		}

		override internal List<Unit> onEventTime(double currentTime)
		{
			unit.moveSpeed -= moveSpeed;
			unit.range -= range;
			unit.damage -= damage;
			return EMPTYLIST;
		}

		override internal bool afterAnotherEvent(Unit affectedUnit, int outcome, double currentTime)
		{
			return unitDead(unit, affectedUnit, outcome);
		}
	}

	public class StealthEvent : Event
	{
		double mana;
		public StealthEvent(Unit unit, double t, double mana) :
			 base(unit, t)
		{
			this.mana = mana;
		}

		override internal bool useAcrossRounds() { return !unit.visible; }

		override internal int getOutcome()
		{
			return NONE;
		}

		override internal List<Unit> onEventTime(double currentTime)
		{
			unit.invisibleBySkill = false;
			unit.visible = true;
			return createListOfUnit();
		}

		override internal bool afterAnotherEvent(Unit affectedUnit, int outcome, double currentTime)
		{
			return !unitAlive(affectedUnit, outcome);
		}
	}

	public class HealthChangeEvent : Event
	{
		int dHealth, range;
		Point targetPos;
		bool hitEnemies;
		public HealthChangeEvent(Point targetPos, double t, int range, int dHealth, bool hitEnemies, Unit user) :
			  base(user, t)
		{
			this.range = range;
			this.targetPos = targetPos;
			this.dHealth = dHealth;
			this.hitEnemies = hitEnemies;
		}

		override internal List<Unit> onEventTime(double currentTime)
		{
			for (int i = Const.game.allUnits.Count - 1; i >= 0; i--)
			{
				Unit target = Const.game.allUnits[i];
				if (unit.team == target.team && hitEnemies) continue;
				if (unit.team != target.team && !hitEnemies) continue;
				if ((target is Tower)) continue;

				double dist2 = targetPos.Distance2(target);
				if (dist2 <= range * range)
				{
					doDamage(target, -1 * dHealth, this.unit);
				}
			}

			return EMPTYLIST;
		}

		override internal bool afterAnotherEvent(Unit affectedUnit, int outcome, double currentTime)
		{
			return false;
		}
	}

	public class ShieldEvent : Event
	{
		int rounds;
		public ShieldEvent(Unit unit, int t) :
			base(unit, t + 1)
		{ // avoid rounding errors.
			rounds = t;
		}

		override internal bool useAcrossRounds()
		{
			rounds--;
			if (rounds <= 0)
			{
				unit.shield = 0;
				return false;
			}

			return true;
		}

		override internal List<Unit> onEventTime(double currentTime)
		{
			unit.shield = 0;
			return EMPTYLIST;
		}

		override internal bool afterAnotherEvent(Unit affectedUnit, int outcome, double currentTime)
		{
			return affectedUnit == this.unit && affectedUnit.shield <= 0;
		}
	}

	public class ExplosiveShieldEvent : Event
	{
		int rounds;
		public ExplosiveShieldEvent(Unit unit, int t) :
			 base(unit, t + 1)
		{
			rounds = t;
		}

		override internal bool useAcrossRounds()
		{
			rounds--;
			if (rounds <= 0)
			{
				unit.explosiveShield = 0;
				return false;
			}

			return true;
		}

		override internal List<Unit> onEventTime(double currentTime)
		{
			unit.explosiveShield = 0;

			return EMPTYLIST;
		}

		override internal bool afterAnotherEvent(Unit affectedUnit, int outcome, double currentTime)
		{
			if (affectedUnit == this.unit && this.unit.explosiveShield <= 0)
			{
				Const.game.events.Add(new ExplosionEvent(affectedUnit, 0, this.unit.x, this.unit.y));

				return true;
			}

			return false;
		}
	}

	public class ExplosionEvent : Event
	{
		double x, y;
		public ExplosionEvent(Unit unit, double t, double x, double y) :
			base(unit, t)
		{
			this.x = x;
			this.y = y;
		}

		override internal List<Unit> onEventTime(double currentTime)
		{
			foreach (Unit target in Const.game.allUnits)
			{
				if (target is Tower || unit.team == target.team) continue;

				double dist2 = unit.Distance2(target);
				if (dist2 <= Const.EXPLOSIVESHIELDRANGE2)
				{
					doDamage(target, Const.EXPLOSIVESHIELDDAMAGE, unit);
				}
			}
			return EMPTYLIST;
		}

		override internal bool afterAnotherEvent(Unit affectedUnit, int outcome, double currentTime)
		{
			return false;
		}
	}

	public class DrainManaEvent : Event
	{
		Hero attacker;
		int manaToDrain;
		public DrainManaEvent(Unit unit, double t, int manaToDrain, Hero attacker) :
			 base(unit, t)
		{
			this.manaToDrain = manaToDrain;
			this.attacker = attacker;
		}

		override internal List<Unit> onEventTime(double currentTime)
		{
			if (unit.isDead) return EMPTYLIST;

			//If attack is dead mana is still drained since spell would be in motion.
			Hero target = (Hero)unit;
			manaToDrain = Math.Min(target.mana, manaToDrain);
			target.mana -= manaToDrain;
			attacker.mana += manaToDrain;
			if (attacker.maxMana < attacker.mana) attacker.mana = attacker.maxMana;
			return EMPTYLIST;
		}

		override internal bool afterAnotherEvent(Unit affectedUnit, int outcome, double currentTime)
		{
			return false;
		}

	}

	public class LineEffectEvent : Event
	{
		MovingEntity movingSpell;
		Hero attacker;
		int damage, damageByTime;
		double radius, duration;
		public LineEffectEvent(Unit unit, double t, MovingEntity movingSpell, int damage, Hero attacker, double radius, double duration, int damageByTime)
		  :
			  base(unit, t)
		{
			this.attacker = attacker;
			this.movingSpell = movingSpell;
			this.damage = damage;
			this.radius = radius;
			this.duration = duration;
			this.damageByTime = damageByTime;
		}

		override internal int getOutcome()
		{
			return NONE;
		}

		override internal List<Unit> onEventTime(double currentTime)
		{
			movingSpell.moveIgnoreEdges(currentTime);
			double dist2 = movingSpell.Distance2(unit);
			double compareDist = Math.Pow(radius, 2);
			if (currentTime <= duration && dist2 <= compareDist)
			{
				doDamage(unit, (int)(damage + damageByTime * currentTime), attacker);
			}
			movingSpell.moveIgnoreEdges(currentTime * -1);

			return EMPTYLIST;
		}

		override internal bool afterAnotherEvent(Unit affectedUnit, int outcome, double currentTime)
		{
			if (currentTime > duration || !unitAlive(affectedUnit, outcome)) return true;

			if (affectedUnit == this.unit && hasAnyOutcome(outcome, SPEEDCHANGED, TELEPORTED))
			{
				movingSpell.moveIgnoreEdges(currentTime);
				double colT = this.unit.getCollisionTime(movingSpell, radius - Const.EPSILON);
				if (colT >= 0)
					t = colT;
				else t = 2;
				movingSpell.moveIgnoreEdges(currentTime * -1);
			}

			return false;
		}
	}

	public class WireEvent : Event
	{
		MovingEntity movingSpell;
		Hero attacker;
		int stun_time;
		double radius, duration, dmgMultiplier;
		List<Hero> potentialTargets;
		public WireEvent(List<Hero> potentialTargets, double t, MovingEntity movingSpell, int stun_time, Hero attacker, double radius, double duration, double dmgMultiplier) :
			 base(null, t)
		{
			this.dmgMultiplier = dmgMultiplier;
			this.attacker = attacker;
			this.movingSpell = movingSpell;
			this.stun_time = stun_time;
			this.radius = radius;
			this.duration = duration;
			this.potentialTargets = potentialTargets;
		}

		override internal int getOutcome()
		{
			return STUNNED;
		}

		override internal List<Unit> onEventTime(double currentTime)
		{
			movingSpell.moveIgnoreEdges(currentTime);
			if (currentTime <= duration)
			{
				Hero closest = null;
				double minDist2 = 0.0;

				foreach (Hero potentialTarget in potentialTargets)
				{
					double dist2 = potentialTarget.Distance2(movingSpell);
					if (dist2 <= Math.Pow(radius + Const.EPSILON, 2) && (closest == null || dist2 < minDist2))
					{
						minDist2 = dist2;
						closest = potentialTarget;
					}
				}

				if (closest != null)
				{
					Const.game.events.Add(new StunEvent(closest, 0, 2));
					doDamage(closest, (int)(closest.maxMana * dmgMultiplier), attacker);
					return createListOfUnit();
				}
			}

			return EMPTYLIST;
		}

		override internal bool afterAnotherEvent(Unit affectedUnit, int outcome, double currentTime)
		{
			if (affectedUnit == null)
			{
				return false;
			}
			if (affectedUnit.team != attacker.team && (affectedUnit is Hero || affectedUnit is Creature) && hasAnyOutcome(outcome, SPEEDCHANGED, TELEPORTED))
			{
				movingSpell.moveIgnoreEdges(currentTime);

				double affectedT = affectedUnit.getCollisionTime(movingSpell, radius);
				if (affectedT < t && !affectedUnit.isDead)
				{
					t = affectedT;
				}
				else
				{
					t = 2;
					foreach (Unit unit in potentialTargets)
					{
						if (unit.isDead) continue;
						double colT = unit.getCollisionTime(movingSpell, radius);
						if (colT >= 0 && colT < t) t = colT;
					}
				}

				movingSpell.moveIgnoreEdges(currentTime * -1);
			}

			return false;
		}
	}

	public class AttackMoveEvent : Event
	{
		Point targetPos;
		public AttackMoveEvent(Unit unit, Point targetPos) :
			base(unit, 1.0)
		{
			this.targetPos = targetPos;

			runSilentlyTowards(unit, targetPos);

			recalculate(null);
		}

		override internal int getOutcome()
		{
			return SPEEDCHANGED;
		}

		override internal List<Unit> onEventTime(double currentTime)
		{

			Unit closest = this.unit.findClosestOnOtherTeam();
			if (unit.canAttack(closest))
			{
				unit.vx = unit.forceVX;
				unit.vy = unit.forceVY;
				unit.fireAttack(closest);
				return createListOfUnit();
			}

			if (currentTime <= 1.0 && closest != null && !closest.isDead) unit.attackUnitOrMoveTowards(closest, currentTime);
			else
			{
				unit.vx = unit.forceVX;
				unit.vy = unit.forceVY;
				return createListOfUnit();
			}
			return EMPTYLIST;
		}

		override internal bool afterAnotherEvent(Unit affectedUnit, int outcome, double currentTime)
		{
			if (unitStopped(this.unit, affectedUnit, outcome))
			{
				return true;
			}
			if (affectedUnit == this.unit || affectedUnit == null || affectedUnit.team != 1 - this.unit.team) return false;
			if (this.unit.Distance2(affectedUnit) <= Const.AGGROUNITRANGE2)
			{
				t = 0.0;
			}
			else if (hasAnyOutcome(outcome, SPEEDCHANGED, TELEPORTED))
			{
				recalculate(affectedUnit);
			}

			return false;
		}

		void recalculate(Unit changedUnit)
		{
			if (changedUnit == null)
			{
				t = 1.0;
				foreach (Unit unit in Const.game.allUnits)
				{
					if (unit.team != 1 - this.unit.team || !this.unit.allowedToAttack(unit)) continue;
					double time = this.unit.getCollisionTime(unit, Const.AGGROUNITRANGE - Const.EPSILON);
					if (time < t && time >= 0)
						t = time;
				}
			}
			else
			{
				double time = this.unit.getCollisionTime(changedUnit, Const.AGGROUNITRANGE - Const.EPSILON);
				if (time < t && time >= 0)
					t = time;
			}
		}
	}

	public class AttackMoveUnitEvent : Event
	{
		Unit nextTarget;
	public	AttackMoveUnitEvent(Unit unit, Unit targetUnit) :
			base(unit, Utilities.timeToReachTarget(unit, targetUnit, unit.moveSpeed))
		{
			this.nextTarget = targetUnit;
			recalculate(true);
		}

		override internal int getOutcome()
		{
			return SPEEDCHANGED;
		}

		override internal List<Unit> onEventTime(double currentTime)
		{
			this.unit.vx = unit.forceVX;
			this.unit.vy = unit.forceVY;

			if (nextTarget.isDead || unitStopped(unit))
			{
				return createListOfUnit();
			}

			unit.fireAttack(nextTarget);
			return createListOfUnit();
		}

		override internal bool afterAnotherEvent(Unit affectedUnit, int outcome, double currentTime)
		{
			if (this.unit != affectedUnit && this.nextTarget != affectedUnit) return false;
			if (unitStopped(this.unit, affectedUnit, outcome))
			{
				t = 0.0;
			}
			else if (nextTarget == affectedUnit && hasOutcome(outcome, TELEPORTED))
			{
				setSpeedAndAlertChange(this.unit, 0, 0); //when target teleports we lose track and just stops.
				return true;
			}
			else if (nextTarget == affectedUnit && hasOutcome(outcome, SPEEDCHANGED))
			{
				recalculate(false);
			}

			return false;
		}

		void recalculate(bool run)
		{
			double prevVx = this.unit.vx;
			double prevVy = this.unit.vy;

			if (unit.canAttack(nextTarget))
			{
				t = 0.0;
			}
			else
			{
				if (run) runSilentlyTowards(this.unit, nextTarget);
				double timeToTarget = this.unit.getCollisionTime(this.nextTarget, this.unit.range - Const.EPSILON);
				if (timeToTarget < 0)
					t = Utilities.timeToReachTarget(unit, nextTarget, unit.moveSpeed);
				else t = timeToTarget;
			}

			if (Math.Abs(this.unit.vx - prevVx) > Const.EPSILON || Math.Abs(this.unit.vy - prevVy) > Const.EPSILON)
			{
				Const.game.events.Add(new SpeedChangedEvent(this.unit, 0.0, this.unit.vx, this.unit.vy)); //Alert speed changed
			}
		}
	}
}