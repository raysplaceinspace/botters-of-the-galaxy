using System;
using System.Collections.Generic;

namespace BOTG_Refree
{
	public enum UnitKilledState
	{
		normal, denied, farmed
	}

	public abstract class Unit : MovingEntity
	{
		public int id;
		public int health;
		public int stunTime;
		public bool visible;
		public bool invisibleBySkill;
		public int team;
		public int damage;
		public int moveSpeed;
		public int range;
		public bool isDead;
		public double attackTime = 0.1;
		public int goldValue;
		public int maxHealth;
		public Player player;
		public int shield = 0;
		public int explosiveShield;
		public bool moving;
		public bool becomingInvis;

		public Unit(double x, double y, int health, int team, int moveSpeed, Player player) : base(x, y, 0, 0)
		{
			id = Const.GLOBAL_ID++;
			this.isDead = false;
			this.team = team;

			this.health = maxHealth = health;
			this.moveSpeed = moveSpeed;
			this.goldValue = 0;
			this.player = player;

			this.invisibleBySkill = false;
			this.visible = true;
			this.stunTime = 0;
		}

		public void adjustShield(int val) {
			//remove min first
			if (shield < explosiveShield) {
				int toRemove = Math.Min(val, shield);
				val -= toRemove;
				shield -= toRemove;
				explosiveShield -= val;
			} else {
				int toRemove = Math.Min(val, explosiveShield);
				val -= toRemove;
				explosiveShield -= toRemove;
				shield -= val;
			}
		}

		public int getShield() {
			return shield + explosiveShield;
		}
		public bool isMelee() { return range <= 150; }

		public abstract String getType();

		public String getPlayerString() {
			return id + " " +
					team + " " +
					getType() + " " +
					(int)x + " " +
					(int)y + " " +
					range + " " +
					health + " " +
					maxHealth + " " +
					getShield() + " " +
					damage + " " +
					moveSpeed + " " +
					stunTime + " " +
					goldValue + " " +
					getExtraProperties();
		}

		protected virtual String getExtraProperties() {
			return "0 0 0 0 0 0 - 1 0";
		}

		void setPoint(Point point) {
			move(point.x, point.y);
		}

		override public void move(double t) {
			if (isDead) return;
			base.move(t);
		}

		internal void moveAttackTowards(Point point) {
			Unit closest = findClosestOnOtherTeam();

			if (canAttack(closest)) {
				fireAttack(closest);
			} else if (closest != null && Distance2(closest) < Const.AGGROUNITRANGE2) {
				Const.game.events.Add(new AttackMoveUnitEvent(this, closest));
			} else Const.game.events.Add(new AttackMoveEvent(this, point));
		}

		internal void attackUnitOrMoveTowards(Unit unit, double t) {
			if (!allowedToAttack(unit)) return;
			if (Distance2(unit) <= range * range) fireAttack(unit);
			else Const.game.events.Add(new AttackMoveUnitEvent(this, unit));
		}

		internal bool allowedToTarget(Unit unit) {
			if (unit == null) return false; // Nothing to see here
			if (isDead) return false; // Codebusters is another game..
			if (stunTime > 0) return false; // Gimme a break
			if (unit.team != team && !unit.visible && !(this is Tower || this is Creature)) return false; // What you see, is what you get
			return true;
		}

		internal bool allowedToAttack(Unit unit) {
			if (!allowedToTarget(unit)) return false;
			if (unit == this) return false; // Can't attack self
			if (unit.isDead || unit.health <= 0) return false; // Dead man tell no tale
			return true;
		}

		internal bool canAttack(Unit unit) {
			if (!allowedToAttack(unit)) return false; // You shall not pass
			if (unit.team == team && unit.health > unit.maxHealth * Const.DENYHEALTH) return false; // Cant deny healthy creep
			if (Distance2(unit) > range * range) return false; // Cant attack far far away
			return true;
		}

		internal void fireAttack(Unit unit)
		{
			if (!canAttack(unit)) return;

			double attackTravelTime = Math.Min(1.0, attackTime + (isMelee() ? 0 : attackTime * Distance(unit) / range));
			if (Const.game.t + attackTravelTime > 1.0)
			{
				return; // no attacks cross rounds.
			}

			Const.game.events.Add(new DamageEvent(unit, this, attackTravelTime, this.damage));
			//Creep aggro.
			if (this is Hero && unit is Hero)
			{
				foreach (Unit u in Const.game.allUnits)
				{
					if (u.team != 1 - team) continue;
					if (u is LaneUnit && Distance2(u) < Const.AGGROUNITRANGE2)
					{
						LaneUnit peasant = ((LaneUnit)u);

						if (peasant.aggroTimeLeft < Const.AGGROUNITTIME)
						{
							peasant.aggroUnit = this;
							peasant.aggroTset = Const.game.t;
						}
						else if (peasant.aggroTset == Const.game.t && peasant.aggroUnit != null && peasant.aggroUnit.Distance2(unit) > this.Distance2(unit))
						{
							peasant.aggroUnit = this;
						}

						peasant.aggroTimeLeft = Const.AGGROUNITTIME;

					}
					if (u is Tower && Distance2(u) < Const.AGGROUNITRANGE2)
					{
						Tower peasant = ((Tower)u);

						if (peasant.aggroTimeLeft < Const.AGGROUNITTIME)
						{
							peasant.aggroUnit = this;
							peasant.aggroTset = Const.game.t;
						}
						else if (peasant.aggroTset == Const.game.t && peasant.aggroUnit != null && peasant.aggroUnit.Distance2(unit) > this.Distance2(unit))
						{
							peasant.aggroUnit = this;
						}

						peasant.aggroTimeLeft = Const.AGGROUNITTIME;
					}
				}
			}
		}

		abstract internal void findAction(List<Unit> allUnits);
		

		internal Unit findClosestOnOtherTeam(string filter = "none") {
			Unit closest = null;
			bool useFilter = filter != "none";
			double minDist = double.MaxValue;
			foreach (Unit unit in Const.game.allUnits)
			{
				if (useFilter && unit.getType() != filter) continue;

				double dist = Distance2(unit);

				//Closest on other team, if equal take lowest health and if equal highest y (to make equal matches)
				if ((unit.team == 1 - team || filter == "GROOT") && allowedToAttack(unit) &&
						(closest == null || dist < minDist || (dist == minDist && (closest.health > unit.health || unit.y > closest.y))))
				//should be //(closest == null || dist < minDist || (dist == minDist && (closest.health > unit.health || (closest.health == unit.health && unit.y > closest.y)))))
				{
					minDist = dist;
					closest = unit;
				}
			}

			return closest;
		}


		internal void runTowards(Point p, double speed) {
			double distance = Distance(p);

			// Avoid a division by zero
			if (Math.Abs(distance) <= Const.EPSILON) {
				return;
			}

			double timeToLocation = Utilities.timeToReachTarget(this, p, speed);

			double coef = (((double)speed)) / distance;
			vx = (p.x - this.x) * coef;
			vy = (p.y - this.y) * coef;
			moving = true;
			if (speed > distance) {
				Const.game.events.Add(new OnLocationReachedEvent(this, timeToLocation));
			}
		}

		internal void runTowards(Point p) {
			runTowards(p, moveSpeed);
		}

		virtual internal void afterRound() {
			if (stunTime > 0) stunTime--;
			x = Utilities.round(x);
			y = Utilities.round(y);
			vx = 0;
			vy = 0;
			moving = false;
		}
	}
}