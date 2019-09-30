using System.Collections.Generic;
using System.Linq;

namespace BOTG_Refree
{
	public enum CreatureState
	{
		peacefull, runningback, aggressive
	}

	public class Creature : Unit
	{

		Point camp;
		public CreatureState state = CreatureState.peacefull;
		public string creatureType;
		public Creature(double x, double y) : base(x, y, 1, -1, 300, null)
		{
			this.camp = new Point(x, y);
		}

		override public string getType()
		{
			return this.creatureType;
		}


		override internal void findAction(List<Unit> allUnits)
		{
			if (isDead || stunTime > 0) return;
			if (this.state == CreatureState.aggressive)
			{
				aggressiveBehavior(allUnits);
			}
			else if (this.state == CreatureState.runningback)
			{
				runningBackBehavior();
			}
		}

		void aggressiveBehavior(List<Unit> allUnits)
		{
			Unit attacker = allUnits
					.Where(u => u is Hero)
					.OrderBy(u => this.Distance2(u))
					.First();
			Unit target = attacker;
			if (Distance2(camp) < Const.AGGROUNITRANGE2)
			{
				this.attackUnitOrMoveTowards(target, 0.0);
			}
			else
			{
				this.state = CreatureState.runningback;
				this.runTowards(camp);
			}
		}

		void runningBackBehavior()
		{
			this.runTowards(camp);
			if (Distance(camp) < 2)
			{
				this.state = CreatureState.peacefull;
				this.health = maxHealth;
			}
		}
	}
}