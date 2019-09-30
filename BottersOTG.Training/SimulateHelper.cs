using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BottersOTG.Intelligence.Decisions;
using BottersOTG.Model;

namespace BottersOTG.Training {
	public static class SimulateHelper {
		public static World Forward(World current, Policy[] teamPolicies) {
			Dictionary<int, GameAction> actions = new Dictionary<int, GameAction>();
			Simulator.AddEnvironmentalActions(current, actions);
			foreach (Unit unit in current.Units) {
				if (unit.UnitType == UnitType.Hero) {
					Tactic tactic = PolicyEvaluator.Evaluate(current, unit, teamPolicies[unit.Team]);
					actions[unit.UnitId] = PolicyEvaluator.TacticToAction(current, unit, tactic);
					// Debug.WriteLine(string.Format("{0}> {1} team {2}: {3}: {4}", current.Tick, unit.HeroType, unit.Team, tactic, CodinGame.Program.FormatHeroAction(actions[unit.UnitId])));
				}
			}
			current = Simulator.Forward(current, actions);
			return current;
		}

	}
}
