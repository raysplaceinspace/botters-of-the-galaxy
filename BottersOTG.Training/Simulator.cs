using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BottersOTG.Model;
using Utils;

namespace BottersOTG.Training {
	public static class Simulator {
		public static void AddEnvironmentalActions(World world, Dictionary<int, GameAction> actions) {
			Unit[] towers = new Unit[2];
			foreach (Unit unit in world.Units) {
				if (unit.UnitType == UnitType.Tower) {
					towers[unit.Team] = unit;
				}
			}
			if (towers[0] == null || towers[1] == null) {
				// Game over
				return;
			}

			foreach (Unit unit in world.Units) {
				GameAction action = null;
				if (unit.UnitType == UnitType.Minion || unit.UnitType == UnitType.Tower) {
					int enemyTeam = World.Enemy(unit.Team);

					if (unit.AggroTicksRemaining > 0) {
						Unit aggroTarget = world.Units.FirstOrDefault(u => u.UnitId == unit.AggroTargetUnitId);
						if (aggroTarget != null) {
							if ((unit.UnitType == UnitType.Minion && unit.Pos.DistanceTo(aggroTarget.Pos) <= World.AggroRange)
								|| (unit.UnitType == UnitType.Tower && unit.Pos.DistanceTo(aggroTarget.Pos) <= unit.AttackRange)) {
								action = new GameAction {
									ActionType = ActionType.Attack,
									UnitId = aggroTarget.UnitId,
								};
							}
						}
					}

					if (action == null) {
						Unit closestEnemy = world.Units.Where(u => u.Team == enemyTeam).MinByOrDefault(enemy => unit.Pos.DistanceTo(enemy.Pos));
						if (closestEnemy != null && unit.Pos.DistanceTo(closestEnemy.Pos) < unit.AttackRange) {
							action = new GameAction {
								ActionType = ActionType.Attack,
								UnitId = closestEnemy.UnitId,
							};
						} else if (unit.UnitType == UnitType.Minion) {
							Unit enemyTower = towers[enemyTeam];
							action = new GameAction {
								ActionType = ActionType.Move,
								Target = new Vector(enemyTower.Pos.X, unit.Pos.Y), // Move in a straight line
							};
						}
					}
				}

				if (action != null) {
					actions[unit.UnitId] = action;
				}
			}
		}

		public static World Forward(World world, Dictionary<int, GameAction> actions) {
			return SimulatorTick.Forward(world, actions);
		}
	}
}
