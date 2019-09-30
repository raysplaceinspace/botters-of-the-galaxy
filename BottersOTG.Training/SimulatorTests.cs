using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BottersOTG.Model;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Utils;

namespace BottersOTG.Training {
	[TestClass]
	public class SimulatorTests {
		[TestMethod]
		public void TestBlink() {
			const int UnitId = 123;
			const int Team = 1;

			Unit initialIronman = WorldGenerator.Ironman(UnitId, Team, new Vector(100, 200));

			World world = new World();
			world.Units.Add(initialIronman);

			Dictionary<int, GameAction> actions = new Dictionary<int, GameAction>();
			actions[UnitId] = new GameAction {
				ActionType = ActionType.Blink,
				Target = new Vector(200, 300),
			};

			World next = Simulator.Forward(world, actions);
			Unit nextIronman = next.Units.Single();
			Assert.AreEqual(200, nextIronman.Pos.X);
			Assert.AreEqual(300, nextIronman.Pos.Y);

			Assert.AreEqual(World.BlinkReplenish - World.BlinkCost, nextIronman.Mana - initialIronman.Mana);
			Assert.AreEqual(World.BlinkCooldown, nextIronman.BlinkCooldown);
		}

		[TestMethod]
		public void TestBlinkOutOfRange() {
			const int UnitId = 123;
			const int Team = 1;

			Unit initialIronman = WorldGenerator.Ironman(UnitId, Team, new Vector(100, 200));

			World world = new World();
			world.Units.Add(initialIronman);

			Dictionary<int, GameAction> actions = new Dictionary<int, GameAction>();
			actions[UnitId] = new GameAction {
				ActionType = ActionType.Blink,
				Target = new Vector(1000, 2000),
			};

			World next = Simulator.Forward(world, actions);
			Unit nextIronman = next.Units.Single();
			Assert.AreEqual(initialIronman.Pos.X, nextIronman.Pos.X);
			Assert.AreEqual(initialIronman.Pos.Y, nextIronman.Pos.Y);
		}

		[TestMethod]
		public void TestFireball() {
			const int Team = 1;

			int unitId = 0;
			Unit initialIronman = WorldGenerator.Ironman(unitId++, Team, new Vector(100, 100));
			Unit initialHulk = WorldGenerator.Hulk(unitId++, World.Enemy(Team), new Vector(500, 500));
			Unit initialDeadpool = WorldGenerator.Deadpool(unitId++, World.Enemy(Team), new Vector(600, 625));

			World world = new World();
			world.Units.Add(initialIronman);
			world.Units.Add(initialHulk);
			world.Units.Add(initialDeadpool);

			Dictionary<int, GameAction> actions = new Dictionary<int, GameAction>();
			Vector target = new Vector(400, 400);
			actions[initialIronman.UnitId] = new GameAction {
				ActionType = ActionType.Fireball,
				Target = target,
			};

			Vector shootDirection = target.Minus(initialIronman.Pos).Unit();

			World next = Simulator.Forward(world, actions);

			Unit nextIronman = next.Units.Single(u => u.HeroType == HeroType.Ironman);
			Assert.AreEqual(World.FireballCost, initialIronman.Mana - nextIronman.Mana);
			Assert.AreEqual(World.FireballCooldown, nextIronman.FireballCooldown);

			Unit nextHulk = next.Units.Single(u => u.HeroType == HeroType.Hulk);
			Vector hulkCollision = Geometry.LinearCollision(initialIronman.Pos, shootDirection, initialHulk.Pos, World.FireballRadius).Value;
			double distanceToHulk = initialIronman.Pos.DistanceTo(hulkCollision);
			Assert.AreEqual(World.FireballDamage(initialIronman.Mana, distanceToHulk), initialHulk.Health - nextHulk.Health, 1.0);

			Unit nextDeadpool = next.Units.Single(u => u.HeroType == HeroType.Deadpool);
			Vector deadpoolCollision = Geometry.LinearCollision(initialIronman.Pos, shootDirection, initialDeadpool.Pos, World.FireballRadius).Value;
			double distanceToDeadpool = initialIronman.Pos.DistanceTo(deadpoolCollision);
			Assert.AreEqual(World.FireballDamage(initialIronman.Mana, distanceToDeadpool), initialDeadpool.Health - nextDeadpool.Health, 1.0);
		}

		[TestMethod]
		public void TestBurning() {
			const int Team = 1;

			int unitId = 0;
			Unit initialIronman = WorldGenerator.Ironman(unitId++, Team, new Vector(100, 100));
			Unit initialHulk = WorldGenerator.Hulk(unitId++, World.Enemy(Team), new Vector(200, 200));
			Unit initialDeadpool = WorldGenerator.Deadpool(unitId++, World.Enemy(Team), new Vector(600, 600));

			World world = new World();
			world.Units.Add(initialIronman);
			world.Units.Add(initialHulk);
			world.Units.Add(initialDeadpool);

			Dictionary<int, GameAction> actions = new Dictionary<int, GameAction>();
			Vector target = new Vector(250, 250);
			actions[initialIronman.UnitId] = new GameAction {
				ActionType = ActionType.Burning,
				Target = target,
			};

			World next = Simulator.Forward(world, actions);

			Unit nextIronman = next.Units.Single(u => u.HeroType == HeroType.Ironman);
			Assert.AreEqual(World.BurningCost, initialIronman.Mana - nextIronman.Mana);
			Assert.AreEqual(World.BurningCooldown, nextIronman.BurningCooldown);

			Unit nextHulk = next.Units.Single(u => u.HeroType == HeroType.Hulk);
			Assert.AreEqual(World.BurningDamage(initialIronman.ManaRegeneration), initialHulk.Health - nextHulk.Health, 1.0);

			Unit nextDeadpool = next.Units.Single(u => u.HeroType == HeroType.Deadpool);
			Assert.AreEqual(nextDeadpool.Health, initialDeadpool.Health);
		}

		[TestMethod]
		public void TestAoeHeal() {
			const int Team = 1;

			int unitId = 0;
			Unit initialDrStrange = WorldGenerator.DoctorStrange(unitId++, Team, new Vector(100, 100));
			Unit initialHulk = WorldGenerator.Hulk(unitId++, Team, new Vector(200, 200));

			initialHulk.Health = 400;

			World world = new World();
			world.Units.Add(initialDrStrange);
			world.Units.Add(initialHulk);

			Dictionary<int, GameAction> actions = new Dictionary<int, GameAction>();
			Vector target = new Vector(250, 250);
			actions[initialDrStrange.UnitId] = new GameAction {
				ActionType = ActionType.AoeHeal,
				Target = target,
			};

			World next = Simulator.Forward(world, actions);

			Unit nextDrStrange = next.Units.Single(u => u.HeroType == HeroType.DoctorStrange);
			Assert.AreEqual(World.AoeHealCost, initialDrStrange.Mana - nextDrStrange.Mana);
			Assert.AreEqual(World.AoeHealCooldown, nextDrStrange.AoeHealCooldown);

			Unit nextHulk = next.Units.Single(u => u.HeroType == HeroType.Hulk);
			Assert.AreEqual(World.AoeHealing(initialDrStrange.Mana), nextHulk.Health - initialHulk.Health, 1.0);
		}

		[TestMethod]
		public void TestShield() {
			const int Team = 1;

			int unitId = 0;
			Unit initialDrStrange = WorldGenerator.DoctorStrange(unitId++, Team, new Vector(100, 100));
			Unit initialHulk = WorldGenerator.Hulk(unitId++, Team, new Vector(200, 200));
			Unit initialDeadpool = WorldGenerator.Deadpool(unitId++, World.Enemy(Team), new Vector(250, 250));

			World world = new World();
			world.Units.Add(initialDrStrange);
			world.Units.Add(initialHulk);
			world.Units.Add(initialDeadpool);

			double shieldBonus = World.ShieldBonus(initialDrStrange.MaxMana);

			// Tick 1
			{
				Dictionary<int, GameAction> actions = new Dictionary<int, GameAction>();
				actions[initialDrStrange.UnitId] = new GameAction {
					ActionType = ActionType.Shield,
					UnitId = initialHulk.UnitId,
				};

				world = Simulator.Forward(world, actions);

				Unit nextDrStrange = world.Units.Single(u => u.HeroType == HeroType.DoctorStrange);
				Assert.AreEqual(World.ShieldCost, initialDrStrange.Mana - nextDrStrange.Mana);
				Assert.AreEqual(World.ShieldCooldown, nextDrStrange.ShieldCooldown);

				Unit nextHulk = world.Units.Single(u => u.HeroType == HeroType.Hulk);
				Assert.AreEqual(shieldBonus, nextHulk.Shield, 1.0);
				Assert.AreEqual(World.ShieldDuration, nextHulk.ShieldTicksRemaining);
			}

			// Tick 2
			{
				Dictionary<int, GameAction> actions = new Dictionary<int, GameAction>();
				actions[initialDeadpool.UnitId] = new GameAction {
					ActionType = ActionType.Attack,
					UnitId = initialHulk.UnitId,
				};

				world = Simulator.Forward(world, actions);

				Unit nextHulk = world.Units.Single(u => u.HeroType == HeroType.Hulk);
				Assert.AreEqual(shieldBonus - initialDeadpool.AttackDamage, nextHulk.Shield);
				Assert.AreEqual(1, nextHulk.ShieldTicksRemaining);
			}
		}

		[TestMethod]
		public void TestPull() {
			const int Team = 1;

			int unitId = 0;
			Unit initialDrStrange = WorldGenerator.DoctorStrange(unitId++, Team, new Vector(100, 0));
			Unit initialIronman = WorldGenerator.Ironman(unitId++, World.Enemy(Team), new Vector(500, 0));

			World world = new World();
			world.Units.Add(initialDrStrange);
			world.Units.Add(initialIronman);

			Dictionary<int, GameAction> actions = new Dictionary<int, GameAction>();
			actions[initialDrStrange.UnitId] = new GameAction {
				ActionType = ActionType.Pull,
				UnitId = initialIronman.UnitId,
			};

			world = Simulator.Forward(world, actions);
			double manaDrain = World.PullDrain(initialIronman.ManaRegeneration);

			Unit nextDrStrange = world.Units.Single(u => u.HeroType == HeroType.DoctorStrange);
			Assert.AreEqual(World.PullCost, initialDrStrange.Mana - nextDrStrange.Mana);
			Assert.AreEqual(World.PullCooldown, nextDrStrange.PullCooldown);

			Unit nextIronman = world.Units.Single(u => u.HeroType == HeroType.Ironman);
			Assert.AreEqual(300.0, nextIronman.Pos.X, 1.0);
			Assert.AreEqual(0.0, nextIronman.Pos.Y, 1.0);
			Assert.AreEqual(manaDrain, initialIronman.Mana - nextIronman.Mana);
		}

		[TestMethod]
		public void TestCounter() {
			const int Team = 1;

			int unitId = 0;
			Unit initialDeadpool = WorldGenerator.Deadpool(unitId++, Team, new Vector(100, 0));
			Unit initialIronman = WorldGenerator.Ironman(unitId++, World.Enemy(Team), new Vector(200, 0));
			Unit initialDrStrange = WorldGenerator.DoctorStrange(unitId++, World.Enemy(Team), new Vector(300, 0));

			initialIronman.Health = 150;

			World world = new World();
			world.Units.Add(initialDeadpool);
			world.Units.Add(initialIronman);
			world.Units.Add(initialDrStrange);

			Dictionary<int, GameAction> actions = new Dictionary<int, GameAction>();
			actions[initialDeadpool.UnitId] = new GameAction { ActionType = ActionType.Counter };
			actions[initialIronman.UnitId] = new GameAction { ActionType = ActionType.Attack, UnitId = initialDeadpool.UnitId };
			actions[initialDrStrange.UnitId] = new GameAction { ActionType = ActionType.Attack, UnitId = initialDeadpool.UnitId };

			world = Simulator.Forward(world, actions);

			Unit nextDeadpool = world.Units.Single(u => u.HeroType == HeroType.Deadpool);
			Assert.AreEqual(World.CounterCost, initialDeadpool.Mana - nextDeadpool.Mana);
			Assert.AreEqual(World.CounterCooldown, nextDeadpool.CounterCooldown);
			Assert.AreEqual(initialDeadpool.Health, nextDeadpool.Health);

			Assert.IsFalse(world.Units.Any(u => u.HeroType == HeroType.Ironman));

			Unit nextDrStrange = world.Units.Single(u => u.HeroType == HeroType.DoctorStrange);
			Assert.AreEqual(initialDrStrange.Health, nextDrStrange.Health);
		}

		[TestMethod]
		public void TestWire() {
			const int Team = 1;

			int unitId = 0;
			Unit initialDeadpool = WorldGenerator.Deadpool(unitId++, Team, new Vector(100, 0));
			Unit initialIronman = WorldGenerator.Ironman(unitId++, World.Enemy(Team), new Vector(200, 0));

			World world = new World();
			world.Units.Add(initialDeadpool);
			world.Units.Add(initialIronman);

			Dictionary<int, GameAction> actions = new Dictionary<int, GameAction>();
			actions[initialDeadpool.UnitId] = new GameAction { ActionType = ActionType.Wire, Target = new Vector(200, 0) };
			actions[initialIronman.UnitId] = new GameAction { ActionType = ActionType.Attack, UnitId = initialDeadpool.UnitId };

			world = Simulator.Forward(world, actions);
			double wireDamage = World.WireDamage(initialIronman.MaxMana);

			Unit nextDeadpool = world.Units.Single(u => u.HeroType == HeroType.Deadpool);
			Assert.AreEqual(World.WireCost, initialDeadpool.Mana - nextDeadpool.Mana);
			Assert.AreEqual(World.WireCooldown, nextDeadpool.WireCooldown);
			Assert.AreEqual(initialDeadpool.Health, nextDeadpool.Health);

			Unit nextIronman = world.Units.Single(u => u.HeroType == HeroType.Ironman);
			Assert.AreEqual(World.WireStunDuration, nextIronman.StunDuration);
			Assert.AreEqual(wireDamage, initialIronman.Health - nextIronman.Health);
		}

		[TestMethod]
		public void TestStealth() {
			const int Team = 1;

			int unitId = 0;
			Unit initialDeadpool = WorldGenerator.Deadpool(unitId++, Team, new Vector(100, 0));
			Unit initialIronman = WorldGenerator.Ironman(unitId++, World.Enemy(Team), new Vector(200, 0));

			World world = new World();
			world.Units.Add(initialDeadpool);
			world.Units.Add(initialIronman);

			{
				Dictionary<int, GameAction> actions = new Dictionary<int, GameAction>();
				actions[initialDeadpool.UnitId] = new GameAction { ActionType = ActionType.Stealth };

				world = Simulator.Forward(world, actions);

				Unit nextDeadpool = world.Units.Single(u => u.HeroType == HeroType.Deadpool);
				Assert.AreEqual(World.StealthCost, initialDeadpool.Mana - nextDeadpool.Mana);
				Assert.AreEqual(World.StealthCooldown, nextDeadpool.StealthCooldown);
				Assert.IsTrue(nextDeadpool.IsVisible);
				Assert.AreEqual(World.StealthDuration + 1, nextDeadpool.StealthTicksRemaining);
			}

			{
				Dictionary<int, GameAction> actions = new Dictionary<int, GameAction>();
				actions[initialIronman.UnitId] = new GameAction { ActionType = ActionType.AttackNearest, UnitId = initialDeadpool.UnitId };

				world = Simulator.Forward(world, actions);

				Unit nextDeadpool = world.Units.Single(u => u.HeroType == HeroType.Deadpool);
				Assert.IsFalse(nextDeadpool.IsVisible);
				Assert.AreEqual(World.StealthDuration, nextDeadpool.StealthTicksRemaining);
				Assert.AreEqual(initialDeadpool.Health, nextDeadpool.Health);
			}
		}

		[TestMethod]
		public void TestCharge() {
			const int Team = 1;

			int unitId = 0;
			Unit initialHulk = WorldGenerator.Hulk(unitId++, Team, new Vector(100, 0));
			Unit initialIronman = WorldGenerator.Ironman(unitId++, World.Enemy(Team), new Vector(400, 0));

			World world = new World();
			world.Units.Add(initialHulk);
			world.Units.Add(initialIronman);

			Dictionary<int, GameAction> actions = new Dictionary<int, GameAction>();
			actions[initialHulk.UnitId] = new GameAction { ActionType = ActionType.Charge, UnitId = initialIronman.UnitId };
			actions[initialIronman.UnitId] = new GameAction { ActionType = ActionType.Move, Target = new Vector(1000, 0) };

			world = Simulator.Forward(world, actions);
			double chargeDamage = World.ChargeDamage(initialHulk.AttackDamage);

			Unit nextHulk = world.Units.Single(u => u.HeroType == HeroType.Hulk);
			Assert.AreEqual(World.ChargeCost, initialHulk.Mana - nextHulk.Mana);
			Assert.AreEqual(World.ChargeCooldown, nextHulk.ChargeCooldown);
			Assert.AreEqual(initialIronman.Pos.X, nextHulk.Pos.X);
			Assert.AreEqual(initialIronman.Pos.Y, nextHulk.Pos.Y);

			Unit nextIronman = world.Units.Single(u => u.HeroType == HeroType.Ironman);
			Assert.AreEqual(chargeDamage + initialHulk.AttackDamage, initialIronman.Health - nextIronman.Health);
			Assert.AreEqual(World.ChargeDuration, nextIronman.ChargeTicksRemaining);
			Assert.AreEqual(initialIronman.MovementSpeed - World.ChargeMovePenalty, nextIronman.Pos.DistanceTo(initialIronman.Pos), 1.0);
		}

		[TestMethod]
		public void TestExplosiveShield() {
			const int Team = 1;

			int unitId = 0;
			Unit initialHulk = WorldGenerator.Hulk(unitId++, Team, new Vector(100, 0));
			Unit initialDeadpool = WorldGenerator.Deadpool(unitId++, World.Enemy(Team), new Vector(200, 0));
			double shieldBonus = World.ExplosiveShieldBonus(initialHulk.MaxHealth);

			World world = new World();
			world.Units.Add(initialHulk);
			world.Units.Add(initialDeadpool);

			// Tick 1 - shield up
			{
				Dictionary<int, GameAction> actions = new Dictionary<int, GameAction>();
				actions[initialHulk.UnitId] = new GameAction { ActionType = ActionType.ExplosiveShield };
				actions[initialDeadpool.UnitId] = new GameAction { ActionType = ActionType.Attack, UnitId = initialHulk.UnitId };
				world = Simulator.Forward(world, actions);

				Unit nextHulk = world.Units.Single(u => u.HeroType == HeroType.Hulk);
				Assert.AreEqual(World.ExplosiveShieldCost, initialHulk.Mana - nextHulk.Mana);
				Assert.AreEqual(World.ExplosiveShieldCooldown, nextHulk.ExplosiveShieldCooldown);

				Assert.AreEqual(shieldBonus - initialDeadpool.AttackDamage, nextHulk.Shield, 1.0);
				Assert.AreEqual(World.ExplosiveShieldDuration, nextHulk.ExplosiveShieldTicksRemaining);
			}

			// Tick 2 - shield explodes
			{
				Dictionary<int, GameAction> actions = new Dictionary<int, GameAction>();
				actions[initialDeadpool.UnitId] = new GameAction { ActionType = ActionType.Attack, UnitId = initialHulk.UnitId };
				world = Simulator.Forward(world, actions);

				Unit nextHulk = world.Units.Single(u => u.HeroType == HeroType.Hulk);
				Assert.AreEqual(0, nextHulk.Shield);

				Unit nextDeadpool = world.Units.Single(u => u.HeroType == HeroType.Deadpool);
				Assert.AreEqual(World.ExplosiveShieldDamage, initialDeadpool.Health - nextDeadpool.Health);
			}
		}

		[TestMethod]
		public void TestBash() {
			const int Team = 1;

			int unitId = 0;
			Unit initialHulk = WorldGenerator.Hulk(unitId++, Team, new Vector(100, 0));
			Unit initialDeadpool = WorldGenerator.Deadpool(unitId++, World.Enemy(Team), new Vector(200, 0));

			World world = new World();
			world.Units.Add(initialHulk);
			world.Units.Add(initialDeadpool);

			Dictionary<int, GameAction> actions = new Dictionary<int, GameAction>();
			actions[initialHulk.UnitId] = new GameAction { ActionType = ActionType.Bash, UnitId = initialDeadpool.UnitId };
			actions[initialDeadpool.UnitId] = new GameAction { ActionType = ActionType.Attack, UnitId = initialHulk.UnitId };
			world = Simulator.Forward(world, actions);

			Unit nextHulk = world.Units.Single(u => u.HeroType == HeroType.Hulk);
			Assert.AreEqual(World.BashCost, initialHulk.Mana - nextHulk.Mana);
			Assert.AreEqual(World.BashCooldown, nextHulk.BashCooldown);
			Assert.AreEqual(initialHulk.Health, nextHulk.Health);
			
			Unit nextDeadpool = world.Units.Single(u => u.HeroType == HeroType.Deadpool);
			Assert.AreEqual(initialHulk.AttackDamage, initialDeadpool.Health - nextDeadpool.Health);
			Assert.AreEqual(World.BashStunDuration, nextDeadpool.StunDuration);
		}

		[TestMethod]
		public void TestSpearFlip() {
			const int Team = 1;

			int unitId = 0;
			Unit initialValkyrie = WorldGenerator.Valkyrie(unitId++, Team, new Vector(200, 200));
			Unit initialDeadpool = WorldGenerator.Deadpool(unitId++, World.Enemy(Team), new Vector(250, 250));
			double spearFlipAttackDamage = World.SpearFlipDamage(initialValkyrie.AttackDamage);

			World world = new World();
			world.Units.Add(initialValkyrie);
			world.Units.Add(initialDeadpool);

			Dictionary<int, GameAction> actions = new Dictionary<int, GameAction>();
			actions[initialValkyrie.UnitId] = new GameAction { ActionType = ActionType.SpearFlip, UnitId = initialDeadpool.UnitId };
			actions[initialDeadpool.UnitId] = new GameAction { ActionType = ActionType.Attack, UnitId = initialValkyrie.UnitId };
			world = Simulator.Forward(world, actions);

			Unit nextValkyrie = world.Units.Single(u => u.HeroType == HeroType.Valkyrie);
			Assert.AreEqual(World.SpearFlipCost, initialValkyrie.Mana - nextValkyrie.Mana);
			Assert.AreEqual(World.SpearFlipCooldown, nextValkyrie.SpearFlipCooldown);
			Assert.AreEqual(initialValkyrie.Health, nextValkyrie.Health);
			
			Unit nextDeadpool = world.Units.Single(u => u.HeroType == HeroType.Deadpool);
			Assert.AreEqual(spearFlipAttackDamage, initialDeadpool.Health - nextDeadpool.Health);
			Assert.AreEqual(World.SpearFlipStunDuration, nextDeadpool.StunDuration);
			Assert.AreEqual(150, nextDeadpool.Pos.X);
			Assert.AreEqual(150, nextDeadpool.Pos.Y);
		}

		[TestMethod]
		public void TestJump() {
			const int Team = 1;

			int unitId = 0;
			Unit initialValkyrie = WorldGenerator.Valkyrie(unitId++, Team, new Vector(200, 200));
			Unit initialDeadpool = WorldGenerator.Deadpool(unitId++, World.Enemy(Team), new Vector(350, 350));

			World world = new World();
			world.Units.Add(initialValkyrie);
			world.Units.Add(initialDeadpool);

			Dictionary<int, GameAction> actions = new Dictionary<int, GameAction>();
			actions[initialValkyrie.UnitId] = new GameAction { ActionType = ActionType.Jump, Target = new Vector(300, 300) };
			world = Simulator.Forward(world, actions);

			Unit nextValkyrie = world.Units.Single(u => u.HeroType == HeroType.Valkyrie);
			Assert.AreEqual(World.JumpCost, initialValkyrie.Mana - nextValkyrie.Mana);
			Assert.AreEqual(World.JumpCooldown, nextValkyrie.JumpCooldown);
			Assert.AreEqual(300, nextValkyrie.Pos.X);
			Assert.AreEqual(300, nextValkyrie.Pos.Y);
			
			Unit nextDeadpool = world.Units.Single(u => u.HeroType == HeroType.Deadpool);
			Assert.AreEqual(initialValkyrie.AttackDamage, initialDeadpool.Health - nextDeadpool.Health);
		}

		[TestMethod]
		public void TestPowerup() {
			const int Team = 1;

			int unitId = 0;
			Unit initialValkyrie = WorldGenerator.Valkyrie(unitId++, Team, new Vector(200, 200));
			Unit initialDeadpool = WorldGenerator.Deadpool(unitId++, World.Enemy(Team), new Vector(300, 300));

			World world = new World();
			world.Units.Add(initialValkyrie);
			world.Units.Add(initialDeadpool);

			{
				Dictionary<int, GameAction> actions = new Dictionary<int, GameAction>();
				actions[initialValkyrie.UnitId] = new GameAction { ActionType = ActionType.Powerup };
				world = Simulator.Forward(world, actions);

				Unit nextValkyrie = world.Units.Single(u => u.HeroType == HeroType.Valkyrie);
				Assert.AreEqual(World.PowerupCost, initialValkyrie.Mana - nextValkyrie.Mana);
				Assert.AreEqual(World.PowerupCooldown, nextValkyrie.PowerupCooldown);
				Assert.AreEqual(World.PowerupDuration, nextValkyrie.PowerupTicksRemaining);
			}

			{
				Dictionary<int, GameAction> actions = new Dictionary<int, GameAction>();
				actions[initialValkyrie.UnitId] = new GameAction { ActionType = ActionType.Attack, UnitId = initialDeadpool.UnitId };
				world = Simulator.Forward(world, actions);

				Unit nextValkyrie = world.Units.Single(u => u.HeroType == HeroType.Valkyrie);
				Assert.AreEqual(World.PowerupDuration - 1, nextValkyrie.PowerupTicksRemaining);

				Unit nextDeadpool = world.Units.Single(u => u.HeroType == HeroType.Deadpool);
				Assert.AreEqual(
					initialValkyrie.AttackDamage + World.PowerupAttackBonus(initialValkyrie.MovementSpeed),
					initialDeadpool.Health - nextDeadpool.Health,
					1.0);
			}
		}
	}
}
