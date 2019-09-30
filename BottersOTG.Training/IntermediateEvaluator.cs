using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BottersOTG.Model;

namespace BottersOTG.Training {
	class IntermediateEvaluator {
		public const double EnemyTeamFactor = 1.0;
		public const double TowerHealthFactor = 0.25;
		public const double HeroHealthFactor = 1.0;
		public const double HeroDamageOutputFactor = 0.0;
		public const double ForwardFactor = 10.0;
		public const double HeroManaFactor = 0.01;
		public const double StunnedFactor = -10.0;
		public const double WinnerFactor = 10000.0;
		public const double NumHeroesFactor = 1000.0;
		public const double MinionHealthFactor = 0.1;
		public const double GoldFactor = 1.0;
		public const double MinionsKilledFactor = 10.0;
		public const double DeniesFactor = 10.0;

		public static Evaluation Evaluate(World world, int myTeam) {
			return new Evaluation {
				MyTeam = EvaluateTeam(world, myTeam),
				EnemyTeam = EvaluateTeam(world, World.Enemy(myTeam)),
			};
		}

		private static TeamEvaluation EvaluateTeam(World world, int team) {
			TeamEvaluation teamEval = new TeamEvaluation();

			teamEval.IsWinner = world.Winner() == team;
			teamEval.Gold = world.Gold[team];
			teamEval.NumMinionsKilled = world.MinionsKilled[team];
			teamEval.NumDenies = world.Denies[team];
			teamEval.HeroDamageOutput = world.HeroDamageOutput[team];

			foreach (Unit unit in world.Units) {
				if (unit.Team == team) {
					if (unit.UnitType == UnitType.Hero) {
						++teamEval.NumHeroes;
						teamEval.HeroHealth += unit.Health;
						teamEval.Mana += unit.Mana;

						double centerX = World.MapWidth / 2;
						double forwardProportion = 1.0 - (Math.Abs(unit.Pos.X - centerX) / (World.MapWidth / 2));
						teamEval.Forward += forwardProportion;

						teamEval.StunnedTicks += unit.StunDuration;
					} else if (unit.UnitType == UnitType.Minion) {
						teamEval.MinionHealth += unit.Health;
					} else if (unit.UnitType == UnitType.Tower) {
						teamEval.TowerHealth += unit.Health;
					}
				}
			}

			teamEval.Score = 
				MinionsKilledFactor * teamEval.NumMinionsKilled +
				DeniesFactor * teamEval.NumDenies +
				ForwardFactor * teamEval.Forward +
				GoldFactor * teamEval.Gold +
				(teamEval.IsWinner ? WinnerFactor : 0) +
				HeroDamageOutputFactor * teamEval.HeroDamageOutput +
				NumHeroesFactor * teamEval.NumHeroes +
				HeroHealthFactor * teamEval.HeroHealth +
				HeroManaFactor * teamEval.Mana +
				StunnedFactor * teamEval.StunnedTicks +
				MinionHealthFactor * teamEval.MinionHealth +
				TowerHealthFactor * teamEval.TowerHealth;
			return teamEval;
		}

		public class Evaluation {
			public TeamEvaluation MyTeam;
			public TeamEvaluation EnemyTeam;

			public double Score {
				get {
					return MyTeam.Score - EnemyTeamFactor * EnemyTeam.Score;
				}
			}

			public override string ToString() {
				return Score.ToString();
			}
		}

		public class TeamEvaluation {
			public bool IsWinner;
			public int Gold;
			public int NumHeroes;
			public int Mana;
			public double HeroDamageOutput;
			public double HeroHealth;
			public double MinionHealth;
			public double TowerHealth;
			public double Forward;
			public int StunnedTicks;
			public int NumMinionsKilled;
			public int NumDenies;

			public double Score;

			public override string ToString() {
				return Score.ToString();
			}
		}

	}
}
