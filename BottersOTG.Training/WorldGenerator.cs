using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BottersOTG.Intelligence.Decisions;
using BottersOTG.Model;
using Utils;

namespace BottersOTG.Training {
	public static class WorldGenerator {
	    public readonly static Vector TowerTeam0 = new Vector(100, 540);
	    public readonly static Vector TowerTeam1 = new Vector(World.MapWidth - TowerTeam0.X, TowerTeam0.Y);

	    public readonly static Vector SpawnTeam0 = new Vector(TowerTeam0.X + 60, TowerTeam0.Y);
	    public readonly static Vector SpawnTeam1 = new Vector(World.MapWidth - SpawnTeam0.X, SpawnTeam0.Y);

	    public readonly static Vector HeroATeam0 = new Vector(TowerTeam0.X + 100, TowerTeam0.Y + 50);
	    public readonly static Vector HeroBTeam0 = new Vector(TowerTeam0.X + 100, TowerTeam0.Y - 50);
	    public readonly static Vector HeroATeam1 = new Vector(World.MapWidth - HeroATeam0.X, HeroATeam0.Y);
	    public readonly static Vector HeroBTeam1 = new Vector(World.MapWidth - HeroATeam0.X, HeroBTeam0.Y);

		public static IEnumerable<Matchup> AllMatchups() {
			foreach (HeroType[] compositionA in AllTeamCompositions()) {
				foreach (HeroType[] compositionB in AllTeamCompositions()) {
					yield return new Matchup {
						Team0 = compositionA,
						Team1 = compositionB,
					};
				}
			}
		}

		public static IEnumerable<HeroType[]> AllTeamCompositions() {
#pragma warning disable 0162
			HeroType[] heroes = EnumUtils.GetEnumValues<HeroType>().Where(h => h != HeroType.None).ToArray();
			if (World.NumHeroesPerTeam == 1) {
				foreach (HeroType hero in heroes) {
					yield return new[] { hero };
				}
			} else if (World.NumHeroesPerTeam == 2) {
				for (int i = 0; i < heroes.Length; ++i) {
					for (int j = i + 1; j < heroes.Length; ++j) {
						yield return new[] { heroes[i], heroes[j] };
					}
				}
			} else {
				throw new InvalidOperationException("Unknown NumHeroesPerTeam: " + World.NumHeroesPerTeam);
			}
#pragma warning restore 0162
		}

		public static World GenerateInitial(HeroType[] heroes0, HeroType[] heroes1) {
			Vector right = new Vector(1, 0);

			World world = new World();
			world.Units.Add(Tower(world.NextUnitId++, 0, TowerTeam0));
			world.Units.Add(Tower(world.NextUnitId++, 1, TowerTeam1));

			world.Units.Add(Hero(world.NextUnitId++, 0, heroes0[0], HeroATeam0));
			world.Units.Add(Hero(world.NextUnitId++, 1, heroes1[0], HeroATeam1));

			if (heroes0.Length >= 2 && heroes1.Length >= 2) {
				world.Units.Add(Hero(world.NextUnitId++, 0, heroes0[1], HeroBTeam0));
				world.Units.Add(Hero(world.NextUnitId++, 1, heroes1[1], HeroBTeam1));
			}

			if (World.SpawnMinions) {
				AddMinions(world);
			}
			if (!World.EnableSpells) {
#pragma warning disable 0162
				foreach (Unit unit in world.Units) {
					unit.Mana = 0;
					unit.MaxMana = 0;
					unit.ManaRegeneration = 0;
				}
#pragma warning restore 0162
			}

			return world;
		}

		public static void AddMinions(World world) {
			AddMinionFormation(world, world.PrimeTeam);
			AddMinionFormation(world, World.Enemy(world.PrimeTeam));
		}

		public static void AddMinionFormation(World world, int team) {
			Vector spawn = team == 0 ? SpawnTeam0 : SpawnTeam1;
			Vector forward = team == 0 ? new Vector(1, 0) : new Vector(-1, 0);
			Vector down = new Vector(0, 1);

			world.Units.Add(MeleeMinion(world.NextUnitId++, team, spawn));
			world.Units.Add(MeleeMinion(world.NextUnitId++, team, spawn.Plus(down.Multiply(50))));
			world.Units.Add(MeleeMinion(world.NextUnitId++, team, spawn.Plus(down.Multiply(100))));

			world.Units.Add(RangedMinion(world.NextUnitId++, team, spawn.Plus(forward.Multiply(-50))));
		}

		public static Unit Tower(int unitId, int team, Vector spawn) {
			return new Unit {
				UnitId = unitId,
				UnitType = UnitType.Tower,
				Team = team,
				Pos = spawn,
				MaxHealth = 3000,
				Health = 3000,
				AttackDamage = 190,
				AttackRange = 400,
			};
		}

		public static Unit MeleeMinion(int unitId, int team, Vector spawn) {
			return new Unit {
				UnitId = unitId,
				UnitType = UnitType.Minion,
				Team = team,
				Pos = spawn,
				MovementSpeed = 150,
				MaxHealth = 400,
				Health = 400,
				AttackDamage = 25,
				AttackRange = 90,
				GoldValue = 30,
			};
		}

		public static Unit RangedMinion(int unitId, int team, Vector spawn) {
			return new Unit {
				UnitId = unitId,
				UnitType = UnitType.Minion,
				Team = team,
				Pos = spawn,
				MovementSpeed = 150,
				MaxHealth = 250,
				Health = 250,
				AttackDamage = 35,
				AttackRange = 300,
				GoldValue = 50,
			};
		}

		public static Unit Hero(int unitId, int team, HeroType heroType, Vector spawn) {
			switch (heroType) {
				case HeroType.Deadpool: return Deadpool(unitId, team, spawn);
				case HeroType.DoctorStrange: return DoctorStrange(unitId, team, spawn);
				case HeroType.Hulk: return Hulk(unitId, team, spawn);
				case HeroType.Ironman: return Ironman(unitId, team, spawn);
				case HeroType.Valkyrie: return Valkyrie(unitId, team, spawn);
				default: throw new ArgumentException("Unknown hero type: " + heroType);
			}
		}

		public static Unit Hulk(int unitId, int team, Vector spawn) {
			return new Unit {
				UnitId = unitId,
				UnitType = UnitType.Hero,
				HeroType = HeroType.Hulk,
				Team = team,
				Pos = spawn,
				MovementSpeed = 200,
				MaxHealth = 1450,
				Health = 1450,
				MaxMana = 90,
				Mana = 90,
				ManaRegeneration = 1,
				AttackDamage = 80,
				AttackRange = 95,
				GoldValue = 300,
			};
		}

		public static Unit Deadpool(int unitId, int team, Vector spawn) {
			return new Unit {
				UnitId = unitId,
				UnitType = UnitType.Hero,
				HeroType = HeroType.Deadpool,
				Team = team,
				Pos = spawn,
				MovementSpeed = 200,
				MaxHealth = 1380,
				Health = 1380,
				MaxMana = 100,
				Mana = 100,
				ManaRegeneration = 1,
				AttackDamage = 80,
				AttackRange = 110,
				GoldValue = 300,
			};
		}

		public static Unit Valkyrie(int unitId, int team, Vector spawn) {
			return new Unit {
				UnitId = unitId,
				UnitType = UnitType.Hero,
				HeroType = HeroType.Valkyrie,
				Team = team,
				Pos = spawn,
				MovementSpeed = 200,
				MaxHealth = 1400,
				Health = 1400,
				MaxMana = 155,
				Mana = 155,
				ManaRegeneration = 2,
				AttackDamage = 65,
				AttackRange = 130,
				GoldValue = 300,
			};
		}

		public static Unit Ironman(int unitId, int team, Vector spawn) {
			return new Unit {
				UnitId = unitId,
				UnitType = UnitType.Hero,
				HeroType = HeroType.Ironman,
				Team = team,
				Pos = spawn,
				MovementSpeed = 200,
				MaxHealth = 820,
				Health = 820,
				MaxMana = 200,
				Mana = 200,
				ManaRegeneration = 2,
				AttackDamage = 60,
				AttackRange = 270,
				GoldValue = 300,
			};
		}

		public static Unit DoctorStrange(int unitId, int team, Vector spawn) {
			return new Unit {
				UnitId = unitId,
				UnitType = UnitType.Hero,
				HeroType = HeroType.DoctorStrange,
				Team = team,
				Pos = spawn,
				MovementSpeed = 200,
				MaxHealth = 955,
				Health = 955,
				MaxMana = 300,
				Mana = 300,
				ManaRegeneration = 2,
				AttackDamage = 50,
				AttackRange = 245,
				GoldValue = 300,
			};
		}
	}
}

