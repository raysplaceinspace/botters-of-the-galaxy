using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BottersOTG.Intelligence.Decisions;
using BottersOTG.Model;
using Utils;

namespace BottersOTG.Training {
	static class RolloutPerformer {
		public static Rollout[] Rollout(Policy policy, Policy adversary) {
			return Winner(policy, adversary, storeRollout: true);
		}

		public static Rollout[] Winner(Policy policy, Policy adversary, bool storeRollout = false) {
			Policy[] teamPolicies = new[] { policy, adversary };

			return WorldGenerator.AllMatchups().AsParallel().Select(matchup => {
				World current = WorldGenerator.GenerateInitial(matchup.Team0, matchup.Team1);

				// Debug.WriteLine("Simulating " + matchup.ToString());
				List<RolloutTick> ticks = storeRollout ? new List<RolloutTick>() : null;
				int? winner = null;
				while (winner == null) {
					Dictionary<int, Tactic> heroTactics = storeRollout ? new Dictionary<int, Tactic>() : null;

					Dictionary<int, GameAction> actions = new Dictionary<int, GameAction>();
					Simulator.AddEnvironmentalActions(current, actions);
					foreach (Unit unit in current.Units) {
						if (unit.UnitType == UnitType.Hero) {
							Tactic tactic = PolicyEvaluator.Evaluate(current, unit, teamPolicies[unit.Team]);
							actions[unit.UnitId] = PolicyEvaluator.TacticToAction(current, unit, tactic);

							heroTactics?.Add(unit.UnitId, tactic);
						}
					}
					ticks?.Add(new RolloutTick {
						HeroTactics = heroTactics,
						World = current,
					});

					current = Simulator.Forward(current, actions);

					winner = current.Winner();
				}

				double winRate;
				if (winner == 0) {
					winRate = 1.0;
				} else if (winner == 1) {
					winRate = 0.0;
				} else {
					winRate = 0.5;
				}
				return new Rollout {
					Matchup = matchup,
					WinRate = winRate,
					Ticks = ticks,
					FinalWorld = storeRollout ? current : null,
					FinalEvaluation = IntermediateEvaluator.Evaluate(current, 0),
				};
			}).ToArray();
		}
	}
}
