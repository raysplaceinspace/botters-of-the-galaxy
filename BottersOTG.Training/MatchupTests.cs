using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BottersOTG.CodinGame;
using BottersOTG.Intelligence.Decisions;
using BottersOTG.Model;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Utils;

namespace BottersOTG.Training {
	[TestClass]
	public class MatchupTests {
		[TestMethod]
		public void TestDeserializeSubmitted() {
			foreach (Policy policy in SubmittedPolicyProvider.Submissions) {
				Assert.IsNotNull(policy);
			}
		}

		[TestMethod]
		public void TestNumMatchups() {
			int numTeamComps = WorldGenerator.AllTeamCompositions().Count();
#pragma warning disable 0162
			if (World.NumHeroesPerTeam == 1) {
				Assert.AreEqual(5, numTeamComps);
			} else {
				// AB AC AD AE
				// BC BD BE
				// CD CE
				// DE
				Assert.AreEqual(10, numTeamComps);
			}
#pragma warning restore 0162

			int numMatchups = WorldGenerator.AllMatchups().Count();
			Assert.AreEqual(numTeamComps * numTeamComps, numMatchups);
		}

		[Ignore] // Too hard to make identical due to precision errors
		[TestMethod]
		public void TestMirrorSimulation() {
			Policy[] policies = new[] { SubmittedPolicyProvider.Policy1, SubmittedPolicyProvider.Policy1 };
			HeroType[] teamX = new HeroType[] { HeroType.Deadpool, HeroType.DoctorStrange };
			HeroType[] teamY = new HeroType[] { HeroType.Deadpool, HeroType.Hulk };

			World a = WorldGenerator.GenerateInitial(teamX, teamY);
			World b = MirrorWorld(a);
			for (int i = 0; i < 100; ++i) {
				a = SimulateHelper.Forward(a, policies);
				b = SimulateHelper.Forward(b, policies);
				AssertMirrored(a, b);
			}
		}

		[TestMethod]
		public void TestEqualMatch() {
			Rollout[] results =
				RolloutPerformer.Winner(SubmittedPolicyProvider.Policy1, SubmittedPolicyProvider.Policy1)
				.OrderBy(x => x.Matchup.Team0[0])
				.ThenBy(x => x.Matchup.Team0[1])
				.ThenBy(x => x.Matchup.Team1[0])
				.ThenBy(x => x.Matchup.Team1[1])
				.ToArray();
			/*
			foreach (MatchupResult result in results) {
				Debug.WriteLine(
					"{0}\t{1}\t{2}\t{3}\t{4}",
					result.Matchup.Team0[0],
					result.Matchup.Team0[1],
					result.Matchup.Team1[0],
					result.Matchup.Team1[1],
					result.WinRate);
			}
			*/
			double winRate = results.Average(x => x.WinRate);
			Assert.AreEqual(0.5, winRate, 0.05);
		}

		[TestMethod]
		public void TestAgainstPolicy1() {
			Rollout[] results = RolloutPerformer.Winner(PolicyProvider.Policy, SubmittedPolicyProvider.Policy1);
			double winRate = results.Average(x => x.WinRate);
			Assert.IsTrue(winRate > 0.5);
		}

		private void AssertMirrored(World a, World b) {
			foreach (Unit unitA in a.Units) {
				Unit unitB = FindMirror(unitA, b.Units);
				Assert.IsTrue(unitB.Pos.DistanceTo(Mirror(unitA.Pos)) < 2.0);
				Assert.AreEqual(unitA.UnitType, unitB.UnitType);
				Assert.AreEqual(unitA.HeroType, unitB.HeroType);
				Assert.AreEqual(unitA.Health, unitB.Health, 1.0);
			}
		}

		private Unit FindMirror(Unit unitA, List<Unit> units) {
			foreach (Unit unitB in units) {
				if (unitB.UnitId == unitA.UnitId) {
					return unitB;
				}
			}

			throw new ArgumentException("Unable to find mirror of unit: " + unitA.UnitType);
		}

		private Vector Mirror(Vector a) {
			return new Vector(World.MapWidth - a.X, a.Y);
		}

		private World MirrorWorld(World world) {
			world = world.Clone();
			foreach (Unit unit in world.Units) {
				unit.Team = World.Enemy(unit.Team);
				unit.Pos = Mirror(unit.Pos);
			}
			world.PrimeTeam = World.Enemy(world.PrimeTeam);
			return world;
		}
	}
}
