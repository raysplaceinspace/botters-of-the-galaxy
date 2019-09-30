using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Utils;
using BottersOTG.Intelligence;
using BottersOTG.Intelligence.Decisions;
using BottersOTG.Model;

namespace Utils {
	public struct Vector {
		public double X;
		public double Y;

		public static Vector Zero {
			get { return new Vector(0, 0); }
		}

		public Vector(double x, double y) {
			X = x;
			Y = y;
		}

		public Vector Plus(Vector a) {
			return new Vector {
				X = X + a.X,
				Y = Y + a.Y,
			};
		}
		
		public Vector Minus(Vector a) {
			return new Vector {
				X = X - a.X,
				Y = Y - a.Y,
			};
		}

		public double Length() {
			return Math.Sqrt(X * X + Y * Y);
		}

		public double DistanceTo(Vector other) {
			return Minus(other).Length();
		}

		public bool InRange(Vector other, int attackRange) {
			return Math.Pow(other.X - X, 2) + Math.Pow(other.Y - Y, 2) <= Math.Pow(attackRange, 2);
		}

		public Vector Multiply(double a) {
			return new Vector {
				X = X * a,
				Y = Y * a,
			};
		}

		public Vector Divide(double a) {
			return new Vector {
				X = X / a,
				Y = Y / a,
			};
		}

		public Vector Unit() {
			return Divide(Length());
		}

		public Vector Reverse() {
			return Multiply(-1);
		}

		public double Dot(Vector a) {
			return X * a.X + Y * a.Y;
		}

		public Vector Towards(Vector other, double maxRange) {
			Vector step = other.Minus(this);
			if (step.Length() > maxRange) {
				step = step.Unit().Multiply(maxRange);
			}
			return Plus(step);
		}

		public override string ToString() {
			return string.Format("{0:0},{1:0}", X, Y);
		}
	}

	public static class Geometry {
		public static Vector? RadialCollision(Vector from, Vector step, double radius) {
			if (step.Length() <= 0) {
				return null;
			}

			var a = step.X * step.X + step.Y * step.Y;
			var b = 2 * from.X * step.X + 2 * from.Y * step.Y;
			var c = from.X * from.X + from.Y * from.Y - radius * radius;

			var determinant = b * b - 4 * a * c;
			if (determinant < 0) {
				return null;
			}

			var d = Math.Sqrt(determinant);
			double stepsToNegative = (-b - d) / (2 * a);
			double stepsToPositive = (-b + d) / (2 * a);

			double numStepsToCollision = stepsToNegative;
			if (numStepsToCollision < 0) {
				numStepsToCollision = stepsToPositive;
			}
			if (numStepsToCollision < 0) {
				return null;
			}

			return from.Plus(step.Multiply(numStepsToCollision));
		}

		public static Vector? LinearCollision(Vector from, Vector step, Vector to, double radius) {
			var distanceTo = to.Minus(from).Dot(step.Unit());
			if (distanceTo < 0) {
				return null;
			} else if (distanceTo == 0) {
				return from;
			}

			var distancePerStep = step.Length();
		    
		    var numSteps = distanceTo / distancePerStep;
		    var finalPosition = from.Plus(step.Multiply(numSteps));
		    var isCollision = to.DistanceTo(finalPosition) < radius;
			if (isCollision) {
				return finalPosition;
			} else {
		        return null;
		    }
		}

		public static Vector RadialNoise(Vector from, double maxRadius, Random random) {
			double angle = 2.0 * Math.PI * random.NextDouble();
			double radius = maxRadius * random.NextDouble();
			return from.Plus(new Vector(Math.Cos(angle), Math.Sin(angle)).Multiply(radius));
		}
	}

	public static class ExtensionMethods {
		public static TValue GetOrDefault<TKey, TValue>(this IDictionary<TKey, TValue> dictionary, TKey key) {
			return dictionary.ContainsKey(key) ? dictionary[key] : default(TValue);
		}

		public static IEnumerable<T> WhereNotNull<T>(this IEnumerable<T> collection) where T : class {
			foreach (T item in collection) {
				if (item != null) {
					yield return item;
				}
			}
		}

		public static T MinBy<T>(this IEnumerable<T> collection, Func<T, double> selector, T defaultValue = default(T)) {
			double bestValue = double.MaxValue;
			T best = defaultValue;
			foreach (T item in collection) {
				double newValue = selector(item);
				if (newValue < bestValue) {
					bestValue = newValue;
					best = item;
				}
			}
			return best;
		}

		public static T MinByOrDefault<T>(this IEnumerable<T> collection, Func<T, double> selector, T defaultValue = default(T)) {
			return MinBy(collection, selector, defaultValue);
		}

		public static T MaxBy<T>(this IEnumerable<T> collection, Func<T, double> selector, T defaultValue = default(T)) {
			double bestValue = -double.MaxValue;
			T best = defaultValue;
			foreach (T item in collection) {
				double newValue = selector(item);
				if (newValue > bestValue) {
					bestValue = newValue;
					best = item;
				}
			}
			return best;
		}

		public static T MaxByOrDefault<T>(this IEnumerable<T> collection, Func<T, double> selector, T defaultValue = default(T)) {
			return MaxBy(collection, selector, defaultValue);
		}
	}

	public static class EnumUtils {
		public static IEnumerable<T> GetEnumValues<T>() {
			return Enum.GetValues(typeof(T)).Cast<T>();
		}

		public static T Parse<T>(string str) {
			return (T)Enum.Parse(typeof(T), str);
		}
	}
}

namespace BottersOTG.Model {
	public enum UnitType {
		Minion,
		Hero,
		Tower,
		Groot,
	}

	public enum HeroType {
		None,
		Deadpool,
		DoctorStrange,
		Hulk,
		Ironman,
		Valkyrie,
	}

	public class Unit {
		public int UnitId;
        public int Team;
        public UnitType UnitType;
        public Vector Pos;
        public int AttackRange;
        public int Health;
        public int MaxHealth;
        public int Shield;
        public int AttackDamage;
        public int MovementSpeed;
        public int StunDuration;
        public int GoldValue;
		public int CountDown1;
	    public int CountDown2;
	    public int CountDown3;
	    public int Mana;
	    public int MaxMana;
	    public int ManaRegeneration;
	    public HeroType HeroType;
	    public bool IsVisible;
	    public int ItemsOwned;

		public List<string> Owned = new List<string>();

		public int BlinkCooldown {
			get { return CountDown1; }
			set { CountDown1 = value; }
		}

		public int FireballCooldown {
			get { return CountDown2; }
			set { CountDown2 = value; }
		}

		public int BurningCooldown {
			get { return CountDown3; }
			set { CountDown3 = value; }
		}

		public int AoeHealCooldown {
			get { return CountDown1; }
			set { CountDown1 = value; }
		}

		public int ShieldCooldown {
			get { return CountDown2; }
			set { CountDown2 = value; }
		}

		public int PullCooldown {
			get { return CountDown3; }
			set { CountDown3 = value; }
		}

		public int CounterCooldown {
			get { return CountDown1; }
			set { CountDown1 = value; }
		}

		public int WireCooldown {
			get { return CountDown2; }
			set { CountDown2 = value; }
		}

		public int StealthCooldown {
			get { return CountDown3; }
			set { CountDown3 = value; }
		}

		public int ChargeCooldown {
			get { return CountDown1; }
			set { CountDown1 = value; }
		}

		public int ExplosiveShieldCooldown {
			get { return CountDown2; }
			set { CountDown2 = value; }
		}
		
		public int BashCooldown {
			get { return CountDown3; }
			set { CountDown3 = value; }
		}

		public int SpearFlipCooldown {
			get { return CountDown1; }
			set { CountDown1 = value; }
		}

		public int JumpCooldown {
			get { return CountDown2; }
			set { CountDown2 = value; }
		}

		public int PowerupCooldown {
			get { return CountDown3; }
			set { CountDown3 = value; }
		}

		public bool IsDetector {
			get { return UnitType == UnitType.Tower || UnitType == UnitType.Groot; }
		}

		// Simulator only
		public int ShieldTicksRemaining = 0;
		public int ExplosiveShieldTicksRemaining = 0;
		public int AggroTargetUnitId = -1;
		public int AggroTicksRemaining = 0;
		public int StealthTicksRemaining = 0;
		public int ChargeTicksRemaining = 0;
		public int PowerupTicksRemaining = 0;

		public Unit Clone() {
			Unit clone = (Unit)MemberwiseClone();
			clone.Owned = clone.Owned.ToList();
			return clone;
		}

		public override string ToString() {
			string type = UnitType == UnitType.Hero ? HeroType.ToString() : UnitType.ToString();
			return string.Format("{0}<{1},{2}>", type, Team, Pos);
		}

		public void Damage(double damage) {
			Shield -= (int)Math.Round(damage);
			if (Shield < 0) {
				Health += Shield;
				Shield = 0;
			}
		}

		public void Heal(double amount) {
			Health += (int)Math.Round(amount);
			if (Health > MaxHealth) {
				Health = MaxHealth;
			}
		}

		public void DrainMana(double amount) {
			Mana -= (int)Math.Round(amount);
			if (Mana < 0) {
				Mana = 0;
			}
		}

		public void Stun(int duration) {
			if (UnitType != UnitType.Tower) {
				StunDuration = Math.Max(StunDuration, duration);
			}
		}
	}

	public class Item {
		public string ItemName;
		public int ItemCost;
        public int Damage;
        public int Health;
        public int MaxHealth;
        public int Mana;
        public int MaxMana;
        public int MoveSpeed;
        public int ManaRegeneration;
        public bool IsPotion;
	}

	public class World {
		public const int NumHeroesPerTeam = 2;
		public const bool SpawnMinions = true;
		public const bool EnableSpells = true;
		public const bool EnableBuySell = true;

		public const int MaxTicks = 250;
		public const int Neutral = -1;
		public const int MapWidth = 1920;
	    public const int MapHeight = 780;
		public const int MaxItems = 4;
		public const double AggroRange = 300;
		public const int SpawnMinionsInterval = 15;

		public const int BlinkCost = 16;
		public const int BlinkReplenish = 20;
		public const int BlinkRange = 200;
		public const int BlinkCooldown = 3;

		public const int FireballCost = 60;
		public const int FireballRange = 900;
		public const int FireballRadius = 50;
		public const int FireballCooldown = 6;

		public const int BurningCost = 50;
		public const int BurningRange = 250;
		public const int BurningRadius = 100;
		public const int BurningCooldown = 5;

		public const int AoeHealCost = 50;
		public const int AoeHealRange = 250;
		public const int AoeHealRadius = 100;
		public const int AoeHealCooldown = 6;

		public const int ShieldCost = 40;
		public const int ShieldDuration = 2;
		public const int ShieldRange = 500;
		public const int ShieldCooldown = 6;

		public const int PullCost = 40;
		public const int PullRange = 400;
		public const int PullDistance = 200;
		public const int PullCooldown = 5;
		public const int PullStunDuration = 1;

		public const int CounterCost = 40;
		public const int CounterRange = 350;
		public const int CounterCooldown = 5;

		public const int WireCost = 50;
		public const int WireRange = 200;
		public const int WireRadius = 50;
		public const int WireStunDuration = 2;
		public const int WireCooldown = 9;

		public const int StealthCost = 30;
		public const int StealthDuration = 5;
		public const int StealthCooldown = 6;

		public const int ChargeCost = 20;
		public const int ChargeRange = 300;
		public const int ChargeCooldown = 4;
		public const int ChargeDuration = 2;
		public const int ChargeMovePenalty = 150;

		public const int ExplosiveShieldCost = 30;
		public const int ExplosiveShieldCooldown = 8;
		public const int ExplosiveShieldDuration = 4;
		public const int ExplosiveShieldRadius = 151;
		public const int ExplosiveShieldDamage = 50;

		public const int BashCost = 40;
		public const int BashCooldown = 10;
		public const int BashRange = 150;
		public const int BashStunDuration = 2;

		public const int SpearFlipCost = 20;
		public const int SpearFlipCooldown = 3;
		public const int SpearFlipRange = 155;
		public const int SpearFlipStunDuration = 1;

		public const int JumpCost = 35;
		public const int JumpCooldown = 3;
		public const int JumpRange = 250;

		public const int PowerupCost = 50;
		public const int PowerupCooldown = 7;
		public const int PowerupDuration = 4;
		public const int PowerupBonusAttackRange = 10;

		public int[] Gold = new int[2];
		public List<Item> Items = new List<Item>();
		public List<Unit> Units = new List<Unit>();

		// Simulator-only
		public int Tick = 0;
		public int[] MinionsKilled = new int[2];
		public int[] Denies = new int[2];
		public double[] HeroDamageOutput = new double[2];
		public int NextUnitId = 0;
		public int PrimeTeam = 0;

		public World Clone() {
			World clone = (World)MemberwiseClone();
			// clone.Items should never change
			clone.Gold = clone.Gold.ToArray();
			clone.Units = clone.Units.Select(u => u.Clone()).ToList();
			clone.MinionsKilled = clone.MinionsKilled.ToArray();
			clone.Denies = clone.Denies.ToArray();
			clone.HeroDamageOutput = HeroDamageOutput.ToArray();
			return clone;
		}

		public int? Winner() {
			int towers0 = Units.Count(u => u.UnitType == UnitType.Tower && u.Team == 0);
			int towers1 = Units.Count(u => u.UnitType == UnitType.Tower && u.Team == 1);
			int heroes0 = Units.Count(u => u.UnitType == UnitType.Hero && u.Team == 0);
			int heroes1 = Units.Count(u => u.UnitType == UnitType.Hero && u.Team == 1);
			bool gameOver = towers0 == 0 || towers1 == 0 || heroes0 == 0 || heroes1 == 0 || Tick >= MaxTicks;

			if (gameOver) {
				int points0 = MinionsKilled[0] + Denies[0];
				int points1 = MinionsKilled[1] + Denies[1];
				if (towers0 > 0 && towers1 == 0) {
					return 0;
				} else if (towers1 > 0 && towers0 == 0) {
					return 1;
				} else if (heroes0 > 0 && heroes1 == 0) {
					return 0;
				} else if (heroes1 > 0 && heroes0 == 0) {
					return 1;
				} else if (points0 > points1) {
					return 0;
				} else if (points1 > points0) {
					return 1;
				} else {
					return World.Neutral; // Draw
				}
			} else {
				return null;
			}
		}

		public static int Enemy(int myTeam) {
			if (myTeam == 0) {
				return 1;
			} else if (myTeam == 1) {
				return 0;
			} else {
				throw new ArgumentException("Team has no opposite: " + myTeam);
			}
		}

		public static double AttackTime(double distance, double attackRange, UnitType unitType) {
			if (distance > attackRange) {
				return 999;
			}

			double baseAttackTime = unitType == UnitType.Hero ? 0.1 : 0.2;
			if (attackRange <= 150) {
				return baseAttackTime;
			} else {
				return baseAttackTime + baseAttackTime * (distance / attackRange);
			}
		}

		public static double FireballDamage(double initialMana, double distanceTravelled) {
			return initialMana * 0.2 + 55.0 * distanceTravelled / 1000.0;
		}

		public static double BurningDamage(double manaRegen) {
			return 5 * manaRegen + 30;
		}

		public static double AoeHealing(int mana) {
			return mana * 0.2;
		}

		public static double ShieldBonus(int maxMana) {
			return maxMana * 0.5 + 50;
		}

		public static double PullDrain(int targetManaRegen) {
			return targetManaRegen * 3 + 5;
		}

		public static double CounterDamage(double damage) {
			return damage * 1.5;
		}

		public static double WireDamage(double targetMaxMana) {
			return targetMaxMana * 0.5;
		}

		public static double ChargeDamage(double attackDamage) {
			return attackDamage * 0.5;
		}

		public static double ExplosiveShieldBonus(double maxHealth) {
			return maxHealth * 0.07 + 50;
		}

		public static double SpearFlipDamage(double attackDamage) {
			return attackDamage * 0.4;
		}

		public static double PowerupAttackBonus(int movementSpeed) {
			return movementSpeed * 0.3;
		}
	}

	public enum ActionType {
		Wait,
		Move,
		Attack,
		AttackNearest,
		AttackMove,
		Buy,
		Sell,

		// Spells ordered by execution priority
		Blink,
		Stealth,

		Wire,

		Shield,
		ExplosiveShield,
		Counter,
		Powerup,

		AoeHeal,

		Fireball,
		Burning,

		// > 0 cast duration
		Charge, // 0.05
		SpearFlip, // 0.1
		Bash, // 0.1 (attacktime)
		Jump, // 0.15
		Pull, // 0.3
	}

	public class GameAction {
		public ActionType ActionType;
		public Vector Target;
		public int UnitId;
		public UnitType UnitType;
		public string ItemName;
		public string Comment;
	}
}

namespace BottersOTG.Intelligence.Decisions {
	public enum Tactic {
		AttackSafely,
		AttackHero,
		Retreat,

		Fireball,
		Burning,
		Blink,

		Wire,
		Stealth,
		Counter,

		Charge,
		ExplosiveShield,
		Bash,

		AoeHeal,
		Shield,
		Pull,

		Jump,
		SpearFlip,
		Powerup,
	}

	public class Policy {
		public HeroChoices HeroMatchups = new HeroChoices();

		public Dictionary<HeroType, IDecisionNode> Root = new Dictionary<HeroType, IDecisionNode>();

		public IDecisionNode Default {
			get { return Root.GetOrDefault(HeroType.None); }
			set { Root[HeroType.None] = value; }
		}

		public IDecisionNode Deadpool {
			get { return Root.GetOrDefault(HeroType.Deadpool); }
			set { Root[HeroType.Deadpool] = value; }
		}

		public IDecisionNode DoctorStrange {
			get { return Root.GetOrDefault(HeroType.DoctorStrange); }
			set { Root[HeroType.DoctorStrange] = value; }
		}

		public IDecisionNode Hulk {
			get { return Root.GetOrDefault(HeroType.Hulk); }
			set { Root[HeroType.Hulk] = value; }
		}

		public IDecisionNode Ironman {
			get { return Root.GetOrDefault(HeroType.Ironman); }
			set { Root[HeroType.Ironman] = value; }
		}

		public IDecisionNode Valkyrie {
			get { return Root.GetOrDefault(HeroType.Valkyrie); }
			set { Root[HeroType.Valkyrie] = value; }
		}

		public Policy Clone() {
			Policy clone = (Policy)MemberwiseClone();
			clone.Root = new Dictionary<HeroType, IDecisionNode>(clone.Root);
			return clone;
		}
	}

	public class HeroChoices {
		public HeroType PrimaryChoice = HeroType.None;
		public Dictionary<HeroType, HeroType> SecondaryChoices = new Dictionary<HeroType, HeroType>();
	}

	public interface IDecisionNode {
		Tactic Evaluate(World world, Unit myHero);
	}

	public class DecisionLeaf : IDecisionNode {
		public readonly Tactic DefaultTactic;
		public readonly Dictionary<HeroType, Tactic> PerHeroTactics;

		public DecisionLeaf(Tactic defaultTactic, Dictionary<HeroType, Tactic> perHeroTactic = null) {
			DefaultTactic = defaultTactic;
			PerHeroTactics = perHeroTactic;
		}

		public Tactic Evaluate(World world, Unit myHero) {
			Tactic tactic;
			if (PerHeroTactics != null && PerHeroTactics.TryGetValue(myHero.HeroType, out tactic)) {
				return tactic;
			} else {
				return DefaultTactic;
			}
		}
	}

	public class DecisionNode : IDecisionNode {
		public IPartitioner Partitioner;

		public IDecisionNode Left;
		public IDecisionNode Right;

		public Tactic Evaluate(World world, Unit myHero) {
			bool goRight = Partitioner.Evaluate(world, myHero);
			if (goRight) {
				return Right.Evaluate(world, myHero);
			} else {
				return Left.Evaluate(world, myHero);
			}
		}
	}

	public interface IPartitioner {
		bool Evaluate(World world, Unit myHero);
	}

	public enum ContinuousAxis {
		MyHealth,
		HealthOfClosestEnemyHero,
		HealthOfClosestOrWeakestEnemyHero,
		HealthOfMyMinions,
		HealthOfMyMinionsInFrontOfMe,
		HealthOfMyMinionsCloseToMe,
		HealthOfEnemyMinions,
		HealthOfEnemyMinionsCloseToMe,
		NetHealthOfMinionsAdvantage,
		NetHealthOfHeroesAdvantage,
		NetNumHeroes,
		MyShield,
		ShieldOfClosestOrWeakestEnemyHero,
		AllyStunDuration,
		StunDurationOfClosestOrWeakestEnemyHero,
		TotalEnemyHeroStunDuration,
		MyMana,
		TicksToKillClosestOrWeakestEnemyHero,
		TicksToDieFromStrongestInRangeHero,
		DistanceToMyTower,
		DistanceToClosestAllyNonHero,
		DistanceToClosestAllyHero,
		DistanceToClosestEnemy,
		DistanceToClosestEnemyHero,
		DistanceToWeakestOrClosestEnemyHero,
		DistanceToFront,
		DistanceToFrontHero,
		MaximumPotentialDamageToMe,
		MaximumPotentialEnemyHeroDamageToMe,
		PotentialAggroDamageToMe,
		PotentialAggroDamageToWeakestOrClosestEnemy,
		TicksToEnemyHeroAttack,
		NetDamageToHeroesAdvantage,
		TicksToMinionSpawn,
	}

	public class ContinuousPartitioner : IPartitioner {
		public readonly ContinuousAxis Axis;
		public readonly double Split;

		public ContinuousPartitioner(ContinuousAxis axis, double split) {
			Axis = axis;
			Split = split;
		}

		public bool Evaluate(World world, Unit myHero) {
			double value = ContinuousAxisEvaluator.Evaluate(Axis, world, myHero);
			return value >= Split;
		}
	}

	public static class ContinuousAxisEvaluator {
		private static double LargeDistance = World.MapHeight + World.MapWidth;
		private static double LargeTicks = 1000;

		public static double Evaluate(ContinuousAxis axis, World world, Unit myHero) {
			switch (axis) {
				case ContinuousAxis.DistanceToClosestAllyNonHero:
					return DistanceToClosest(world, myHero, u => u.Team == myHero.Team && u.UnitType != UnitType.Hero);
				case ContinuousAxis.DistanceToClosestEnemy:
					return DistanceToClosest(world, myHero, u => u.Team == World.Enemy(myHero.Team));
				case ContinuousAxis.DistanceToClosestEnemyHero:
					return DistanceToClosest(world, myHero, u => u.Team == World.Enemy(myHero.Team) && u.UnitType == UnitType.Hero);
				case ContinuousAxis.DistanceToWeakestOrClosestEnemyHero: {
					Unit enemyHero = DecisionUtils.WeakestOrClosestEnemyHero(world, myHero);
					return enemyHero == null ? LargeDistance : myHero.Pos.DistanceTo(enemyHero.Pos);
				}
				case ContinuousAxis.DistanceToClosestAllyHero:
					return DistanceToClosest(world, myHero, u => u.Team == myHero.Team && u.UnitType == UnitType.Hero && u.UnitId != myHero.UnitId);
				case ContinuousAxis.DistanceToMyTower:
					return DistanceToClosest(world, myHero, u => u.Team == myHero.Team && u.UnitType == UnitType.Tower);
				case ContinuousAxis.DistanceToFront:
					return DistanceToFront(world, myHero, UnitType.Minion);
				case ContinuousAxis.DistanceToFrontHero:
					return DistanceToFront(world, myHero, UnitType.Hero);
				case ContinuousAxis.MyHealth:
					return myHero.Health;
				case ContinuousAxis.MyMana:
					return myHero.Mana;
				case ContinuousAxis.MyShield:
					return myHero.Shield;
				case ContinuousAxis.AllyStunDuration:
					return world.Units.FirstOrDefault(u => u.Team == myHero.Team && u.UnitType == UnitType.Hero)?.StunDuration ?? 0;
				case ContinuousAxis.StunDurationOfClosestOrWeakestEnemyHero:
					return DecisionUtils.WeakestOrClosestEnemyHero(world, myHero)?.StunDuration ?? 0;
				case ContinuousAxis.TotalEnemyHeroStunDuration:
					return world.Units.Where(u => u.Team == World.Enemy(myHero.Team) && u.UnitType == UnitType.Hero).Sum(u => u.StunDuration);
				case ContinuousAxis.HealthOfClosestEnemyHero:
					return DecisionUtils.ClosestOrDefault(world, myHero, u => u.Team == World.Enemy(myHero.Team) && u.UnitType == UnitType.Hero)?.Health ?? 0;
				case ContinuousAxis.ShieldOfClosestOrWeakestEnemyHero:
					return DecisionUtils.ClosestOrDefault(world, myHero, u => u.Team == World.Enemy(myHero.Team) && u.UnitType == UnitType.Hero)?.Shield ?? 0;
				case ContinuousAxis.HealthOfClosestOrWeakestEnemyHero:
					return DecisionUtils.WeakestOrClosestEnemyHero(world, myHero)?.Health ?? 0;
				case ContinuousAxis.HealthOfMyMinions:
					return world.Units.Where(u => u.UnitType == UnitType.Minion && u.Team == myHero.Team).Sum(u => u.Health);
				case ContinuousAxis.HealthOfMyMinionsInFrontOfMe: {
					IEnumerable<Unit> minions = world.Units.Where(u => u.UnitType == UnitType.Minion && u.Team == myHero.Team);
					if (myHero.Team == 0) {
						return minions.Where(u => u.Pos.X > myHero.Pos.X).Sum(u => u.Health);
					} else {
						return minions.Where(u => u.Pos.X < myHero.Pos.X).Sum(u => u.Health);
					}
				}
				case ContinuousAxis.HealthOfMyMinionsCloseToMe:
					return world.Units.Where(u => u.Team == myHero.Team && u.UnitType == UnitType.Minion && u.Pos.DistanceTo(myHero.Pos) <= 300).Sum(u => u.Health);
				case ContinuousAxis.HealthOfEnemyMinionsCloseToMe:
					return world.Units.Where(u => u.Team == World.Enemy(myHero.Team) && u.UnitType == UnitType.Minion && u.Pos.DistanceTo(myHero.Pos) <= 300).Sum(u => u.Health);
				case ContinuousAxis.HealthOfEnemyMinions:
					return world.Units.Where(u => u.UnitType == UnitType.Minion && u.Team == World.Enemy(myHero.Team)).Sum(u => u.Health);
				case ContinuousAxis.NetHealthOfMinionsAdvantage: 
					return world.Units.Where(u => u.UnitType == UnitType.Minion).Sum(u => u.Health * (u.Team == myHero.Team ? 1 : -1));
				case ContinuousAxis.NetHealthOfHeroesAdvantage: 
					return world.Units.Where(u => u.UnitType == UnitType.Hero).Sum(u => u.Health * (u.Team == myHero.Team ? 1 : -1));
				case ContinuousAxis.NetNumHeroes: 
					return world.Units.Where(u => u.UnitType == UnitType.Hero).Sum(u => u.Team == myHero.Team ? 1 : -1);
				case ContinuousAxis.TicksToKillClosestOrWeakestEnemyHero:
					return (DecisionUtils.WeakestOrClosestEnemyHero(world, myHero)?.Health ?? 0) / myHero.AttackDamage;
				case ContinuousAxis.TicksToDieFromStrongestInRangeHero: {
					Unit strongestInRangeEnemyHero =
						world.Units
						.Where(u => u.Team == World.Enemy(myHero.Team) && u.UnitType == UnitType.Hero && u.Pos.DistanceTo(myHero.Pos) <= u.AttackRange)
						.MaxByOrDefault(u => u.AttackDamage);
					return strongestInRangeEnemyHero == null ? LargeTicks : myHero.Health / strongestInRangeEnemyHero.AttackDamage;
				}
				case ContinuousAxis.MaximumPotentialDamageToMe:
					return world.Units.Where(u => u.Team == World.Enemy(myHero.Team) && u.Pos.DistanceTo(myHero.Pos) <= u.AttackRange).Sum(u => u.AttackDamage);
				case ContinuousAxis.MaximumPotentialEnemyHeroDamageToMe:
					return world.Units.Where(u => u.Team == World.Enemy(myHero.Team) && u.UnitType == UnitType.Hero && u.Pos.DistanceTo(myHero.Pos) <= u.AttackRange).Sum(u => u.AttackDamage);
				case ContinuousAxis.PotentialAggroDamageToMe:
					return AggroDamageTo(world, myHero);
				case ContinuousAxis.PotentialAggroDamageToWeakestOrClosestEnemy:
					return AggroDamageTo(world, DecisionUtils.WeakestOrClosestEnemyHero(world, myHero));
				case ContinuousAxis.TicksToEnemyHeroAttack:
					return
						world.Units.Where(u => u.Team == World.Enemy(myHero.Team) && u.UnitType == UnitType.Hero)
						.Select(u => Math.Max(0, u.Pos.DistanceTo(myHero.Pos) - u.AttackRange) / u.MovementSpeed)
						.DefaultIfEmpty().Min();
				case ContinuousAxis.NetDamageToHeroesAdvantage:
					return NetDamageToHeroesAdvantage(world, myHero.Team);
				case ContinuousAxis.TicksToMinionSpawn:
					return World.SpawnMinionsInterval - (world.Tick % World.SpawnMinionsInterval);
				default: throw new ArgumentException("Unknown axis: " + axis);
			}
		}

		private static double AggroDamageTo(World world, Unit targetHero) {
			return targetHero == null ? 0 : world.Units.Where(u => u.Team == World.Enemy(targetHero.Team) && InAggroRange(u, targetHero)).Sum(u => u.AttackDamage);
		}

		private static bool InAggroRange(Unit fromUnit, Unit toUnit) {
			return
				fromUnit.UnitType == UnitType.Minion && fromUnit.Pos.DistanceTo(toUnit.Pos) <= World.AggroRange
				|| fromUnit.UnitType == UnitType.Tower && fromUnit.Pos.DistanceTo(toUnit.Pos) <= fromUnit.AttackRange;
		}

		private static double DistanceToFront(World world, Unit myHero, UnitType unitType) {
			double forwardDirection = myHero.Team == 0 ? 1 : -1;
			Unit front = 
				world.Units
				.Where(u => u.Team == myHero.Team)
				.Where(u => u.UnitType == unitType || u.UnitType == UnitType.Tower)
				.MaxByOrDefault(u => u.Pos.X * forwardDirection);
			if (front == null) {
				// We're dead already
				return 0;
			}

			return forwardDirection * (front.Pos.X - myHero.Pos.X);
		}

		private static double DistanceToClosest(World world, Unit myHero, Func<Unit, bool> predicate) {
			return
				world.Units
				.Where(predicate)
				.Select(u => myHero.Pos.DistanceTo(u.Pos))
				.DefaultIfEmpty(LargeDistance)
				.Min();
		}

		private static double NetDamageToHeroesAdvantage(World world, int myTeam) {
			List<Unit> heroes = world.Units.Where(u => u.UnitType == UnitType.Hero).ToList();
			double netDamage = 0.0;
			foreach (Unit unit in world.Units) {
				if (unit.Team == World.Neutral) {
					continue;
				}
				bool canAttackHero = heroes.Any(hero => hero.Team == World.Enemy(unit.Team) && unit.Pos.DistanceTo(hero.Pos) < unit.AttackRange);
				if (canAttackHero) {
					if (unit.Team == myTeam) {
						netDamage += unit.AttackDamage;
					} else {
						netDamage -= unit.AttackDamage;
					}
				}
			}
			return netDamage;
		}
	}

	public enum CategoricalBoolean {
		False,
		True,
	}

	public enum CategoricalAxis {
		MyHero,
		MyHeroes,
		EnemyHeroes,
		IsVisible,
		MySpells,
		AllySpells,
		EnemySpells,
	}

	public class CategoricalPartitioner : IPartitioner {
		public readonly CategoricalAxis Axis;
		public readonly HashSet<Enum> Categories;

		public CategoricalPartitioner(CategoricalAxis axis, Enum[] categories) {
			Axis = axis;
			Categories = new HashSet<Enum>(categories);
		}

		public bool Evaluate(World world, Unit myHero) {
			return CategoricalAxisEvaluator.Evaluate(Axis, world, myHero).Any(category => Categories.Contains(category));
		}
	}

	public static class CategoricalAxisEvaluator {
		public static IEnumerable<Enum> Evaluate(CategoricalAxis axis, World world, Unit myHero) {
			switch (axis) {
				case CategoricalAxis.MyHero:
					return new Enum[] { myHero.HeroType };
				case CategoricalAxis.MyHeroes:
					return Heroes(world, myHero.Team).Select(u => u.HeroType).Cast<Enum>();
				case CategoricalAxis.EnemyHeroes:
					return Heroes(world, World.Enemy(myHero.Team)).Select(u => u.HeroType).Cast<Enum>();
				case CategoricalAxis.IsVisible:
					return myHero.IsVisible ? new Enum[] { CategoricalBoolean.True } : new Enum[0];
				case CategoricalAxis.MySpells:
					return AvailableSpells(myHero).Cast<Enum>();
				case CategoricalAxis.AllySpells: {
					Unit ally = world.Units.FirstOrDefault(u => u.UnitType == UnitType.Hero && u.Team == myHero.Team && u.UnitId != myHero.UnitId);
					return ally == null ? new Enum[0] : AvailableSpells(ally).Cast<Enum>();
				}
				case CategoricalAxis.EnemySpells:
					return EnemySpells(world, myHero).Cast<Enum>();
				default: throw new ArgumentException("Unknown CategoricalAxis: " + axis);
			}
		}

		private static IEnumerable<Unit> Heroes(World world, int team) {
			return world.Units.Where(u => u.UnitType == UnitType.Hero && u.Team == team);
		}

		private static IEnumerable<ActionType> AvailableSpells(Unit hero, Vector? target = null) {
			if (hero.HeroType == HeroType.Deadpool) {
				if (hero.CounterCooldown == 0 && hero.Mana >= World.CounterCost) {
					yield return ActionType.Counter;
				}
				if (hero.WireCooldown == 0 && hero.Mana >= World.WireCost && InRange(hero, target, World.WireRange)) {
					yield return ActionType.Wire;
				}
				if (hero.StealthCooldown == 0 && hero.Mana >= World.StealthCost) {
					yield return ActionType.Stealth;
				}
			} else if (hero.HeroType == HeroType.DoctorStrange) {
				if (hero.AoeHealCooldown == 0 && hero.Mana >= World.AoeHealCost && InRange(hero, target, World.AoeHealRange)) {
					yield return ActionType.AoeHeal;
				}
				if (hero.ShieldCooldown == 0 && hero.Mana >= World.ShieldCost && InRange(hero, target, World.ShieldRange)) {
					yield return ActionType.Shield;
				}
				if (hero.PullCooldown == 0 && hero.Mana >= World.PullCost && InRange(hero, target, World.PullRange)) {
					yield return ActionType.Pull;
				}
			} else if (hero.HeroType == HeroType.Hulk) {
				if (hero.ChargeCooldown == 0 && hero.Mana >= World.ChargeCost && InRange(hero, target, World.ChargeRange)) {
					yield return ActionType.Charge;
				}
				if (hero.ExplosiveShieldCooldown == 0 && hero.Mana >= World.ExplosiveShieldCost) {
					yield return ActionType.ExplosiveShield;
				}
				if (hero.BashCooldown == 0 && hero.Mana >= World.BashCost && InRange(hero, target, World.BashRange)) {
					yield return ActionType.Bash;
				}
			} else if (hero.HeroType == HeroType.Ironman) {
				if (hero.BlinkCooldown == 0 && hero.Mana >= World.BlinkCost && InRange(hero, target, World.BlinkRange)) {
					yield return ActionType.Blink;
				}
				if (hero.FireballCooldown == 0 && hero.Mana >= World.FireballCost && InRange(hero, target, World.FireballRange)) {
					yield return ActionType.Fireball;
				}
				if (hero.BurningCooldown == 0 && hero.Mana >= World.BurningCost && InRange(hero, target, World.BurningRange)) {
					yield return ActionType.Burning;
				}
			} else if (hero.HeroType == HeroType.Valkyrie) {
				if (hero.SpearFlipCooldown == 0 && hero.Mana >= World.SpearFlipCost && InRange(hero, target, World.SpearFlipRange)) {
					yield return ActionType.SpearFlip;
				}
				if (hero.JumpCooldown == 0 && hero.Mana >= World.JumpCost && InRange(hero, target, World.JumpRange)) {
					yield return ActionType.Jump;
				}
				if (hero.PowerupCooldown == 0 && hero.Mana >= World.PowerupCost) {
					yield return ActionType.Powerup;
				}
			}
		}

		private static bool InRange(Unit hero, Vector? target, int maxRange) {
			if (target.HasValue) {
				return hero.Pos.InRange(target.Value, maxRange);
			} else {
				return true;
			}
		}

		private static IEnumerable<ActionType> EnemySpells(World world, Unit myHero) {
			return world.Units.Where(u => u.UnitType == UnitType.Hero && u.Team == World.Enemy(myHero.Team)).SelectMany(u => AvailableSpells(u, myHero.Pos));
		}
	}

	public static class PolicyEvaluator {
		public static Tactic Evaluate(World world, Unit myHero, Policy policy) {
			return (policy.Root.GetOrDefault(myHero.HeroType) ?? policy.Default)?.Evaluate(world, myHero) ?? Tactic.AttackSafely;
		}

		public static GameAction TacticToAction(World world, Unit myHero, Tactic tactic) {
			switch (tactic) {
				case Tactic.Retreat: return Retreat(world, myHero);
				case Tactic.AttackSafely: return AttackSafely(world, myHero);
				case Tactic.AttackHero: return AttackHero(world, myHero);

				case Tactic.Blink: return Blink(world, myHero);
				case Tactic.Fireball: return Fireball(world, myHero);
				case Tactic.Burning: return Burning(world, myHero);

				case Tactic.Counter: return Counter(world, myHero);
				case Tactic.Wire: return Wire(world, myHero);
				case Tactic.Stealth: return Stealth(world, myHero);

				case Tactic.Charge: return Charge(world, myHero);
				case Tactic.ExplosiveShield: return ExplosiveShield(world, myHero);
				case Tactic.Bash: return Bash(world, myHero);

				case Tactic.AoeHeal: return AoeHeal(world, myHero);
				case Tactic.Shield: return Shield(world, myHero);
				case Tactic.Pull: return Pull(world, myHero);

				case Tactic.SpearFlip: return SpearFlip(world, myHero);
				case Tactic.Jump: return Jump(world, myHero);
				case Tactic.Powerup: return Powerup(world, myHero);
				default: throw new ArgumentException("Unknown tactic: " + tactic);
			}
		}

		private static GameAction Retreat(World world, Unit myHero) {
			const int SafeBuffer = 1;

			int myTeam = myHero.Team;
			Unit myTower = world.Units.FirstOrDefault(u => u.UnitType == UnitType.Tower && u.Team == myTeam);
			Unit enemyTower = world.Units.FirstOrDefault(u => u.UnitType == UnitType.Tower && u.Team == World.Enemy(myTeam));
			if (myTower == null || enemyTower == null) {
				return new GameAction { ActionType = ActionType.Wait };
			}

			Vector forward = enemyTower.Pos.Minus(myTower.Pos).Unit();
			Vector retreatTarget = myTower.Pos.Plus(forward.Multiply(-SafeBuffer)); // behind tower;

			if (World.EnableSpells && myHero.HeroType == HeroType.Ironman && myHero.Mana >= World.BlinkCost && myHero.BlinkCooldown == 0) {
				return new GameAction {
					ActionType = ActionType.Blink,
					Target = myHero.Pos.Towards(retreatTarget, World.BlinkRange - SafeBuffer),
					Comment = "Blink Retreat",
				};
			} else {
				return new GameAction {
					ActionType = ActionType.Move,
					Target = retreatTarget,
					Comment = "Retreating",
				};
			}
		}

		private static GameAction Deny(World world, Unit myHero) {
			int myTeam = myHero.Team;
			int enemyTeam = World.Enemy(myTeam);

			List<Unit> myUnits = world.Units.Where(u => u.Team == myTeam && u.UnitType == UnitType.Minion).ToList();
			List<Unit> enemyUnits = world.Units.Where(u => u.Team == enemyTeam).ToList();
			List<Unit> myDeadUnits = new List<Unit>();
			foreach (Unit unit in myUnits) {
				IEnumerable<Unit> killers = enemyUnits.Where(enemy => enemy.Pos.InRange(unit.Pos, enemy.AttackRange));
				if (unit.Health < killers.Sum(u => u.AttackDamage)) {
					myDeadUnits.Add(unit);
				}
			}

			Unit denyTarget = myDeadUnits.Where(u => u.Health <= myHero.AttackDamage && u.Pos.InRange(myHero.Pos, myHero.AttackRange)).MaxByOrDefault(u => u.GoldValue);
			if (denyTarget != null) {
				return new GameAction {
					ActionType = ActionType.Attack,
					UnitId = denyTarget.UnitId,
					Comment = "Denying",
				};
			}

			return null;
		}

		private static GameAction AttackSafely(World world, Unit myHero) {
			const double SafeBuffer = 1;

			int myTeam = myHero.Team;
			int enemyTeam = World.Enemy(myTeam);

			Unit myTower = world.Units.FirstOrDefault(u => u.UnitType == UnitType.Tower && u.Team == myTeam);
			Unit enemyTower = world.Units.FirstOrDefault(u => u.UnitType == UnitType.Tower && u.Team == enemyTeam);
			if (myTower == null || enemyTower == null) {
				return new GameAction { ActionType = ActionType.Wait };
			}
			Vector forward = enemyTower.Pos.Minus(myTower.Pos).Unit();

			/*
			IEnumerable<Unit> heroesThatCanHitMe = world.Units.Where(u => u.UnitType == UnitType.Hero && u.Team == World.Enemy(myHero.Team) && u.Pos.DistanceTo(myHero.Pos) <= u.AttackRange);
			IEnumerable<Unit> heroThreats = heroesThatCanHitMe.Where(u => myTower.Pos.DistanceTo(u.Pos) > myTower.AttackRange);
			if (heroThreats.Any()) {
				return new Model.GameAction {
					ActionType = ActionType.Move,
					Target = myTower.Pos.Plus(forward.Multiply(-SafeBuffer)),
					Comment = "Repositioning",
				};
			}
			*/

			List<Unit> myMinions = world.Units.Where(u => u.Team == myTeam).ToList();
			List<Unit> enemyUnits = world.Units.Where(u => u.Team == enemyTeam).ToList();
			List<Unit> mySafeUnits = new List<Unit>();
			List<Unit> myDeadUnits = new List<Unit>();
			foreach (Unit unit in myMinions) {
				IEnumerable<Unit> killers = enemyUnits.Where(enemy => enemy.Pos.DistanceTo(unit.Pos) <= enemy.AttackRange);
				if (unit.Health < killers.Sum(u => u.AttackDamage)) {
					myDeadUnits.Add(unit);
				} else {
					mySafeUnits.Add(unit);
				}
			}

			double enemyFrontRange = enemyUnits.Where(u => u.UnitType != UnitType.Hero).Select(u => u.Pos.Minus(myTower.Pos).Dot(forward)).DefaultIfEmpty().Min();
			double frontierRange = Math.Max(0, Math.Min(enemyFrontRange, mySafeUnits.Where(u => u.UnitType != UnitType.Hero).Select(u => u.Pos.Minus(myTower.Pos).Dot(forward)).DefaultIfEmpty().Max()));
			if (myHero.Pos.Minus(myTower.Pos).Dot(forward) >= frontierRange) {
				return new GameAction {
					ActionType = ActionType.Move,
					Target = myTower.Pos.Plus(forward.Multiply(-SafeBuffer)),
					Comment = "Overextended",
				};
			}

			/*
			List<Unit> deniableUnits = myDeadUnits.Where(u => u.Pos.DistanceTo(myHero.Pos) < myHero.AttackRange).ToList();
			if (deniableUnits.Count > 0) {
				Unit target = deniableUnits.MaxBy(u => u.GoldValue);
				return new GameAction {
					ActionType = ActionType.Attack,
					UnitId = target.UnitId,
					Comment = "Denying",
				};
			}
			*/

			double frontierAttackRange = frontierRange + myHero.AttackRange;
			List<Unit> vulnerableEnemies = world.Units.Where(u => u.Team == enemyTeam && u.IsVisible && u.Pos.DistanceTo(myTower.Pos) <= frontierAttackRange).ToList();
			if (vulnerableEnemies.Count > 0) {
				Unit target = vulnerableEnemies.MinBy(u => u.Health);
				return new GameAction {
					ActionType = ActionType.Attack,
					UnitId = target.UnitId,
					Comment = "Attacking",
				};
			}

			GameAction itemAction = ChooseItemAction(world, myHero);
			if (itemAction != null) {
				return itemAction;
			}

			GameAction denyAction = Deny(world, myHero);
			if (denyAction != null) {
				return denyAction;
			}

			Vector behindFront = myTower.Pos.Plus(forward.Multiply(frontierRange - SafeBuffer));
			return new GameAction {
				ActionType = ActionType.Move,
				Target = behindFront,
				Comment = "Forward",
			};
		}

		private static GameAction ChooseItemAction(World world, Unit myHero) {
			if (!World.EnableBuySell) {
				return null;
			}

			const double HealPercent = 0.4;
			int itemLimit = World.MaxItems - 1; // one slot open for a health potion

			int myGold = world.Gold[myHero.Team];

			Dictionary<string, Item> itemLookup = world.Items.ToDictionary(x => x.ItemName);
			List<Item> owned = myHero.Owned.Select(itemName => itemLookup[itemName]).ToList();

			List<Item> healthPotions = world.Items.Where(x => x.IsPotion && x.Health > 0 && x.ItemCost <= myGold).ToList();
			if (myHero.ItemsOwned <= itemLimit &&
				healthPotions.Count > 0 &&
				myHero.Health < (HealPercent * myHero.MaxHealth)) {
				double maxHeal = myHero.MaxHealth - myHero.Health;
				Item healthPotion = healthPotions.MaxBy(x => Math.Min(maxHeal, x.Health) / x.ItemCost);
				return new GameAction {
					ActionType = ActionType.Buy,
					ItemName = healthPotion.ItemName,
					Comment = "Using health potion",
				};
			}

			List<Item> damageIncreasers = world.Items.Where(x => x.Damage > 0 && x.ItemCost <= myGold).ToList();
			if (damageIncreasers.Count > 0) {
				Item toAdd = damageIncreasers.MaxBy(x => x.Damage);
				Item toRemove = null;
				if (myHero.ItemsOwned >= itemLimit) {
					toRemove = owned.MinBy(x => x.Damage); // Assume only have damage increasers
				}

				double netGain = toAdd.Damage - (toRemove?.Damage ?? 0);
				if (netGain > 0) {
					if (toRemove != null) {
						return new GameAction {
							ActionType = ActionType.Sell,
							ItemName = toRemove.ItemName,
							Comment = "Selling " + toRemove.ItemName,
						};
					} else {
						return new GameAction {
							ActionType = ActionType.Buy,
							ItemName = toAdd.ItemName,
							Comment = "Buying " + toAdd.ItemName,
						};
					}
				}
			}

			return null;
		}

		private static GameAction AttackHero(World world, Unit myHero) {
			int myTeam = myHero.Team;
			int enemyTeam = World.Enemy(myTeam);

			List<Unit> enemyHeroes = world.Units.Where(u => u.UnitType == UnitType.Hero && u.Team == enemyTeam).ToList();
			if (enemyHeroes.Count == 0) {
				return new GameAction {
					ActionType = ActionType.AttackNearest,
					UnitType = UnitType.Hero,
					Comment = "Advancing",
				};
			}

			Unit weakestEnemyHeroInRange = enemyHeroes.Where(enemyHero => myHero.Pos.DistanceTo(enemyHero.Pos) <= myHero.AttackRange).MinByOrDefault(u => u.Health);
			if (weakestEnemyHeroInRange != null) {
				return new GameAction {
					ActionType = ActionType.Attack,
					UnitId = weakestEnemyHeroInRange.UnitId,
					Comment = "Assassinating",
				};
			}

			Unit nearestEnemyHero = enemyHeroes.MinBy(enemyHero => myHero.Pos.DistanceTo(enemyHero.Pos));
			return new GameAction {
				ActionType = ActionType.Attack,
				UnitId = nearestEnemyHero.UnitId,
				Comment = "Fighting",
			};
		}

		private static GameAction Powerup(World world, Unit myHero) {
			if (myHero.HeroType != HeroType.Valkyrie || myHero.Mana < World.PowerupCost || myHero.PowerupCooldown > 0) {
				return AttackSafely(world, myHero);
			}

			return new GameAction { ActionType = ActionType.Powerup, Comment = "Powerup" };
		}

		private static GameAction Jump(World world, Unit myHero) {
			if (myHero.HeroType != HeroType.Valkyrie || myHero.Mana < World.JumpCost || myHero.JumpCooldown > 0) {
				return AttackSafely(world, myHero);
			}

			Unit target = DecisionUtils.WeakestEnemyHero(world, myHero, World.JumpRange);
			if (target == null) {
				return AttackSafely(world, myHero);
			}

			return new GameAction {
				ActionType = ActionType.Jump,
				Target = target.Pos,
				Comment = "Jump",
			};
		}

		private static GameAction SpearFlip(World world, Unit myHero) {
			if (myHero.HeroType != HeroType.Valkyrie || myHero.Mana < World.SpearFlipCost || myHero.SpearFlipCooldown > 0) {
				return AttackSafely(world, myHero);
			}

			Unit target = DecisionUtils.WeakestEnemyHero(world, myHero, World.SpearFlipRange);
			if (target == null) {
				return AttackSafely(world, myHero);
			}

			return new GameAction {
				ActionType = ActionType.SpearFlip,
				UnitId = target.UnitId,
				Comment = "SpearFlip",
			};
		}

		private static GameAction Pull(World world, Unit myHero) {
			if (myHero.HeroType != HeroType.DoctorStrange || myHero.Mana < World.PullCost || myHero.PullCooldown > 0) {
				return AttackSafely(world, myHero);
			}

			Unit target = DecisionUtils.WeakestEnemyHero(world, myHero, World.PullRange);
			if (target == null) {
				return AttackSafely(world, myHero);
			}

			return new GameAction {
				ActionType = ActionType.Pull,
				UnitId = target.UnitId,
				Comment = "Pull",
			};
		}

		private static GameAction Shield(World world, Unit myHero) {
			if (myHero.HeroType != HeroType.DoctorStrange || myHero.Mana < World.ShieldCost || myHero.ShieldCooldown > 0) {
				return AttackSafely(world, myHero);
			}

			Unit target =
				world.Units
				.Where(u => u.Team == myHero.Team && u.Pos.DistanceTo(myHero.Pos) <= World.ShieldRange)
				.MinBy(u => u.Health);
			if (target == null || target.Health == target.MaxHealth || target.Shield > 0) {
				return AttackSafely(world, myHero);
			}

			return new GameAction {
				ActionType = ActionType.Shield,
				UnitId = target.UnitId,
				Comment = "Shield",
			};
		}

		private static GameAction AoeHeal(World world, Unit myHero) {
			if (myHero.HeroType != HeroType.DoctorStrange || myHero.Mana < World.AoeHealCost || myHero.AoeHealCooldown > 0) {
				return AttackSafely(world, myHero);
			}

			Unit target =
				world.Units
				.Where(u => u.Team == myHero.Team && u.Pos.DistanceTo(myHero.Pos) <= World.AoeHealRange)
				.MinBy(u => u.Health);
			if (target == null || target.Health == target.MaxHealth) {
				return AttackSafely(world, myHero);
			}

			return new GameAction {
				ActionType = ActionType.AoeHeal,
				Target = target.Pos,
				Comment = "AoeHeal",
			};
		}

		private static GameAction Bash(World world, Unit myHero) {
			if (myHero.HeroType != HeroType.Hulk || myHero.Mana < World.BashCost || myHero.BashCooldown > 0) {
				return AttackSafely(world, myHero);
			}

			Unit target = DecisionUtils.WeakestEnemyHero(world, myHero, World.BashRange);
			if (target == null) {
				return AttackSafely(world, myHero);
			}

			return new GameAction {
				ActionType = ActionType.Bash,
				UnitId = target.UnitId,
				Comment = "Bash",
			};

		}

		private static GameAction ExplosiveShield(World world, Unit myHero) {
			if (myHero.HeroType != HeroType.Hulk || myHero.Mana < World.ExplosiveShieldCost || myHero.ExplosiveShieldCooldown > 0) {
				return AttackSafely(world, myHero);
			}

			return new GameAction { ActionType = ActionType.ExplosiveShield, Comment = "ExplosiveShield" };
		}

		private static GameAction Charge(World world, Unit myHero) {
			if (myHero.HeroType != HeroType.Hulk || myHero.Mana < World.ChargeCost || myHero.ChargeCooldown > 0) {
				return AttackSafely(world, myHero);
			}

			Unit target = DecisionUtils.WeakestEnemyHero(world, myHero, World.ChargeRange);
			if (target == null) {
				return AttackSafely(world, myHero);
			}

			return new GameAction {
				ActionType = ActionType.Charge,
				UnitId = target.UnitId,
				Comment = "Charge",
			};
		}

		private static GameAction Stealth(World world, Unit myHero) {
			if (myHero.HeroType != HeroType.Deadpool || myHero.Mana < World.StealthCost || myHero.StealthCooldown > 0) {
				return AttackSafely(world, myHero);
			}

			Unit target = DecisionUtils.WeakestOrClosestEnemyHero(world, myHero, myHero.AttackRange);
			if (target == null) {
				return AttackSafely(world, myHero);
			}

			if (myHero.Pos.DistanceTo(target.Pos) <= myHero.AttackRange) {
				return new GameAction {
					ActionType = ActionType.Stealth,
					Target = myHero.Pos,
					Comment = "Vanishing",
				};
			} else {
				return new GameAction {
					ActionType = ActionType.Stealth,
					Target = target.Pos,
					Comment = "Stealthing",
				};
			}
		}

		private static GameAction Wire(World world, Unit myHero) {
			if (myHero.HeroType != HeroType.Deadpool || myHero.Mana < World.WireCost || myHero.WireCooldown > 0) {
				return AttackSafely(world, myHero);
			}

			Unit target = DecisionUtils.WeakestEnemyHero(world, myHero, World.WireRange);
			if (target == null) {
				return AttackSafely(world, myHero);
			}

			return new GameAction {
				ActionType = ActionType.Wire,
				Target = target.Pos,
				Comment = "Wire",
			};

		}

		private static GameAction Counter(World world, Unit myHero) {
			if (myHero.HeroType != HeroType.Deadpool || myHero.Mana < World.CounterCost || myHero.CounterCooldown > 0) {
				return AttackSafely(world, myHero);
			}

			return new GameAction { ActionType = ActionType.Counter, Comment = "Counter" };

		}

		private static GameAction Burning(World world, Unit myHero) {
			if (myHero.HeroType != HeroType.Ironman || myHero.Mana < World.BurningCost || myHero.BurningCooldown > 0) {
				return AttackSafely(world, myHero);
			}

			Unit target = DecisionUtils.WeakestEnemyHero(world, myHero, World.BurningRange);
			if (target == null) {
				return AttackSafely(world, myHero);
			}

			return new GameAction {
				ActionType = ActionType.Burning,
				Target = target.Pos,
				Comment = "Burning",
			};
		}

		private static GameAction Fireball(World world, Unit myHero) {
			if (myHero.HeroType != HeroType.Ironman || myHero.Mana < World.FireballCost || myHero.FireballCooldown > 0) {
				return AttackSafely(world, myHero);
			}

			Unit target = DecisionUtils.WeakestEnemyHero(world, myHero, World.FireballRange);
			if (target == null) {
				return AttackSafely(world, myHero);
			}

			return new GameAction {
				ActionType = ActionType.Fireball,
				Target = target.Pos,
				Comment = "Fireball",
			};
		}

		private static GameAction Blink(World world, Unit myHero) {
			if (myHero.HeroType != HeroType.Ironman || myHero.Mana < World.BlinkCost || myHero.BlinkCooldown > 0) {
				return AttackSafely(world, myHero);
			}

			return new GameAction {
				ActionType = ActionType.Blink,
				Target = myHero.Pos,
				Comment = "Blink",
			};
		}
	}

	public static class DecisionUtils {
		public static Unit WeakestOrClosestEnemyHero(World world, Unit myHero, double? attackRange = null) {
			return 
				WeakestEnemyHero(world, myHero, attackRange) ??
				ClosestOrDefault(world, myHero, u => u.UnitType == UnitType.Hero && u.Team == World.Enemy(myHero.Team));
		}

		public static Unit WeakestEnemyHero(World world, Unit myHero, double? attackRange = null) {
			return
				world.Units
				.Where(u => u.UnitType == UnitType.Hero && u.Team == World.Enemy(myHero.Team) && u.Pos.DistanceTo(myHero.Pos) < (attackRange ?? myHero.AttackRange))
				.MinByOrDefault(u => u.Health);
		}

		public static Unit ClosestOrDefault(World world, Unit myHero, Func<Unit, Boolean> predicate) {
			return
				world.Units
				.Where(predicate)
				.MinByOrDefault(u => myHero.Pos.DistanceTo(u.Pos));
		}
	}

}

namespace BottersOTG.Intelligence {
	public class Agent {
		public static HeroType ChooseHero(IEnumerable<HeroType> enemyHeroes, List<HeroType> alreadyChosen, Policy policy, Random random) {
			if (alreadyChosen.Count == 0) {
				if (policy.HeroMatchups.PrimaryChoice != HeroType.None) {
					return policy.HeroMatchups.PrimaryChoice;
				}
			} else if (alreadyChosen.Count == 1) {
				HeroType enemyHero = enemyHeroes.FirstOrDefault();
				HeroType secondaryHero = policy.HeroMatchups.SecondaryChoices.GetOrDefault(enemyHero);
				if (secondaryHero != HeroType.None) {
					return secondaryHero;
				}
			}

			HeroType[] options = EnumUtils.GetEnumValues<HeroType>().Except(alreadyChosen.Concat(new[] { HeroType.None })).ToArray();
			return options[random.Next(options.Length)];
		}

		public static GameAction ChooseAction(World world, Unit myHero, int myTeam, Policy policy) {
			Tactic tactic = PolicyEvaluator.Evaluate(world, myHero, policy);
			return PolicyEvaluator.TacticToAction(world, myHero, tactic);
		}
	}
}

namespace BottersOTG.CodinGame {
	public class Program {
		public static void Main(string[] args) {
			Random random = new Random();
			List<HeroType> alreadyChosen = new List<HeroType>();

			Dictionary<int, List<string>> ownedItems = new Dictionary<int, List<string>>();

	        string[] inputs;
	        int myTeam = int.Parse(Console.ReadLine());
	        int bushAndSpawnPointCount = int.Parse(Console.ReadLine()); // usefrul from wood1, represents the number of bushes and the number of places where neutral units can spawn
	        for (int i = 0; i < bushAndSpawnPointCount; i++)
	        {
	            inputs = Console.ReadLine().Split(' ');
	            string entityType = inputs[0]; // BUSH, from wood1 it can also be SPAWN
	            int x = int.Parse(inputs[1]);
	            int y = int.Parse(inputs[2]);
	            int radius = int.Parse(inputs[3]);
	        }

			List<Item> items = new List<Item>();
	        int itemCount = int.Parse(Console.ReadLine()); // useful from wood2
	        for (int i = 0; i < itemCount; i++)
	        {
	            inputs = Console.ReadLine().Split(' ');

				items.Add(new Item {
		            ItemName = inputs[0],
		            ItemCost = int.Parse(inputs[1]),
		            Damage = int.Parse(inputs[2]),
		            Health = int.Parse(inputs[3]),
		            MaxHealth = int.Parse(inputs[4]),
		            Mana = int.Parse(inputs[5]),
		            MaxMana = int.Parse(inputs[6]),
		            MoveSpeed = int.Parse(inputs[7]),
		            ManaRegeneration = int.Parse(inputs[8]),
		            IsPotion = int.Parse(inputs[9]) == 1 ? true : false,
				});
	        }

			// game loop
			int tick = 0;
	        while (true)
	        {
				World world = new World() {
					Items = items,
				};

	            world.Gold[myTeam] = int.Parse(Console.ReadLine());
	            world.Gold[World.Enemy(myTeam)] = int.Parse(Console.ReadLine());

	            int roundType = int.Parse(Console.ReadLine()); // a positive value will show the number of heroes that await a command
	            int entityCount = int.Parse(Console.ReadLine());
	            for (int i = 0; i < entityCount; i++)
	            {
	                inputs = Console.ReadLine().Split(' ');
					Unit unit = new Unit {
						UnitId = int.Parse(inputs[0]),
						Team = int.Parse(inputs[1]),
						UnitType = ParseUnitType(inputs[2]),
						Pos = new Vector(int.Parse(inputs[3]), int.Parse(inputs[4])),
						AttackRange = int.Parse(inputs[5]),
						Health = int.Parse(inputs[6]),
						MaxHealth = int.Parse(inputs[7]),
						Shield = int.Parse(inputs[8]),
						AttackDamage = int.Parse(inputs[9]),
						MovementSpeed = int.Parse(inputs[10]),
						StunDuration = int.Parse(inputs[11]),
						GoldValue = int.Parse(inputs[12]),
						CountDown1 = int.Parse(inputs[13]),
						CountDown2 = int.Parse(inputs[14]),
						CountDown3 = int.Parse(inputs[15]),
						Mana = int.Parse(inputs[16]),
						MaxMana = int.Parse(inputs[17]),
						ManaRegeneration = int.Parse(inputs[18]),
						HeroType = ParseHeroType(inputs[19]),
						IsVisible = int.Parse(inputs[20]) == 1 ? true : false,
						ItemsOwned = int.Parse(inputs[21]),
					};
					List<string> owned;
					if (ownedItems.TryGetValue(unit.UnitId, out owned)) {
						unit.Owned = owned;
					}
					world.Units.Add(unit);
	            }

	            // Write an action using Console.WriteLine()
	            // To debug: Console.Error.WriteLine("Debug messages...");


	            // If roundType has a negative value then you need to output a Hero name, such as "DEADPOOL" or "VALKYRIE".
	            // Else you need to output roundType number of any valid action, such as "WAIT" or "ATTACK unitId"
	            if (roundType < 0) {
					IEnumerable<HeroType> enemyHeroes =
						world.Units
						.Where(u => u.UnitType == UnitType.Hero && u.Team == World.Enemy(myTeam))
						.Select(u => u.HeroType);
					HeroType nextHero = Agent.ChooseHero(enemyHeroes, alreadyChosen, PolicyProvider.Policy, random);
	                Console.WriteLine(FormatHeroType(nextHero));
					alreadyChosen.Add(nextHero);
	            } else {
					world.Tick = tick++;

					foreach (Unit myHero in world.Units.Where(u => u.UnitType == UnitType.Hero && u.Team == myTeam)) {
						GameAction action = Agent.ChooseAction(world, myHero, myHero.Team, PolicyProvider.Policy);
						Console.WriteLine(FormatHeroAction(action));

						if (action.ActionType == ActionType.Buy || action.ActionType == ActionType.Sell) {
							List<string> owned;
							if (ownedItems.TryGetValue(myHero.UnitId, out owned)) {
								owned = owned.ToList();
							} else {
								owned = new List<string>();
							}

							if (action.ActionType == ActionType.Buy) {
								if (!items.First(item => item.ItemName == action.ItemName).IsPotion) {
									owned.Add(action.ItemName);
								}
							} else if (action.ActionType == ActionType.Sell) {
								owned.Remove(action.ItemName);
							}
							ownedItems[myHero.UnitId] = owned;
						}
					}
	            }
	        }
		}

		public static UnitType ParseUnitType(string unitTypeStr) {
			switch (unitTypeStr) {
				case "GROOT": return UnitType.Groot;
				case "HERO": return UnitType.Hero;
				case "TOWER": return UnitType.Tower;
				case "UNIT": return UnitType.Minion;
				default: throw new ArgumentException("Unknown unit type: " + unitTypeStr);
			}
		}

		public static HeroType ParseHeroType(string heroTypeStr) {
			switch (heroTypeStr) {
				case "-": return HeroType.None;
				case "DEADPOOL": return HeroType.Deadpool;
				case "DOCTOR_STRANGE": return HeroType.DoctorStrange;
				case "HULK": return HeroType.Hulk;
				case "IRONMAN": return HeroType.Ironman;
				case "VALKYRIE": return HeroType.Valkyrie;
				default: return HeroType.None;
			}
		}

		public static string FormatHeroType(HeroType heroType) {
			switch (heroType) {
				case HeroType.Deadpool: return "DEADPOOL";
				case HeroType.DoctorStrange: return "DOCTOR_STRANGE";
				case HeroType.Hulk: return "HULK";
				case HeroType.Ironman: return "IRONMAN";
				case HeroType.Valkyrie: return "VALKYRIE";
				default: return heroType.ToString();
			}
		}

		public static string FormatHeroAction(GameAction action) {
			switch (action.ActionType) {
				case ActionType.Attack: return string.Format("ATTACK {0};{1}", action.UnitId, action.Comment);
				case ActionType.AttackMove: return string.Format("MOVE_ATTACK {0:0} {1:0} {2};{3}", action.Target.X, action.Target.Y, action.UnitId, action.Comment);
				case ActionType.AttackNearest: return string.Format("ATTACK_NEAREST {0};{1}", action.UnitType, action.Comment);
				case ActionType.Move: return string.Format("MOVE {0:0} {1:0};{2}", action.Target.X, action.Target.Y, action.Comment);
				case ActionType.Buy: return string.Format("BUY {0};{1}", action.ItemName, action.Comment);
				case ActionType.Sell: return string.Format("SELL {0};{1}", action.ItemName, action.Comment);
				case ActionType.Counter: return string.Format("COUNTER;{0}", action.Comment);
				case ActionType.Wire: return string.Format("WIRE {0:0} {1:0};{2}", action.Target.X, action.Target.Y, action.Comment);
				case ActionType.Stealth: return string.Format("STEALTH {0:0} {1:0};{2}", action.Target.X, action.Target.Y, action.Comment);
				case ActionType.AoeHeal: return string.Format("AOEHEAL {0:0} {1:0};{2}", action.Target.X, action.Target.Y, action.Comment);
				case ActionType.Shield: return string.Format("SHIELD {0};{1}", action.UnitId, action.Comment);
				case ActionType.Pull: return string.Format("PULL {0};{1}", action.UnitId, action.Comment);
				case ActionType.Charge: return string.Format("CHARGE {0};{1}", action.UnitId, action.Comment);
				case ActionType.ExplosiveShield: return string.Format("EXPLOSIVESHIELD;{0}", action.Comment);
				case ActionType.Bash: return string.Format("BASH {0};{1}", action.UnitId, action.Comment);
				case ActionType.Blink: return string.Format("BLINK {0:0} {1:0};{2}", action.Target.X, action.Target.Y, action.Comment);
				case ActionType.Fireball: return string.Format("FIREBALL {0:0} {1:0};{2}", action.Target.X, action.Target.Y, action.Comment);
				case ActionType.Burning: return string.Format("BURNING {0:0} {1:0};{2}", action.Target.X, action.Target.Y, action.Comment);
				case ActionType.SpearFlip: return string.Format("SPEARFLIP {0};{1}", action.UnitId, action.Comment);
				case ActionType.Jump: return string.Format("JUMP {0:0} {1:0};{2}", action.Target.X, action.Target.Y, action.Comment);
				case ActionType.Powerup: return string.Format("POWERUP;{0}", action.Comment);
				default: return string.Format("WAIT;{0}", action.Comment);
			}
		}
	}

	public static class PolicyProvider {
		public static DecisionNode N(IPartitioner partitioner, IDecisionNode left, IDecisionNode right) {
			return new DecisionNode() {
				Partitioner = partitioner,
				Left = left,
				Right = right,
			};
		}

		public static DecisionLeaf L(string tacticName, string d = null, string h = null, string i = null, string s = null, string v = null) {
			Dictionary<HeroType, Tactic> heroTactics = new Dictionary<HeroType, Tactic>();
			if (d != null) {
				heroTactics[HeroType.Deadpool] = EnumUtils.Parse<Tactic>(d);
			}
			if (h != null) {
				heroTactics[HeroType.Hulk] = EnumUtils.Parse<Tactic>(h);
			}
			if (i != null) {
				heroTactics[HeroType.Ironman] = EnumUtils.Parse<Tactic>(i);
			}
			if (s != null) {
				heroTactics[HeroType.DoctorStrange] = EnumUtils.Parse<Tactic>(s);
			}
			if (v != null) {
				heroTactics[HeroType.Valkyrie] = EnumUtils.Parse<Tactic>(v);
			}
			return new DecisionLeaf(EnumUtils.Parse<Tactic>(tacticName), heroTactics);
		}

		public static ContinuousPartitioner D(string axisName, double split) {
			return new ContinuousPartitioner(EnumUtils.Parse<ContinuousAxis>(axisName), split);
		}

		public static CategoricalPartitioner C(string axisName, params Enum[] categories) {
			return new CategoricalPartitioner(EnumUtils.Parse<CategoricalAxis>(axisName), categories);
		}

		public static Policy Policy =
new Policy {
	HeroMatchups = new HeroChoices {
		PrimaryChoice = HeroType.Ironman,
		SecondaryChoices = new Dictionary<HeroType, HeroType> {
			{ HeroType.Deadpool, HeroType.Valkyrie },
			{ HeroType.Hulk, HeroType.Hulk },
			{ HeroType.DoctorStrange, HeroType.Valkyrie },
			{ HeroType.Ironman, HeroType.Valkyrie },
			{ HeroType.Valkyrie, HeroType.Valkyrie },
		},
	},
	Default = null,
	Deadpool = N(D("DistanceToFront", 105.16),
		N(D("MaximumPotentialDamageToMe", 190.0),
			N(D("DistanceToClosestAllyNonHero", 140.299),
				N(D("HealthOfMyMinionsInFrontOfMe", 30.0),
					N(C("EnemySpells", ActionType.Wire, ActionType.ExplosiveShield, ActionType.Bash, ActionType.Jump),
						N(D("TicksToEnemyHeroAttack", 2.048),
							N(D("DistanceToClosestEnemyHero", 108.628),
								L("AttackHero"),
								N(D("DistanceToClosestEnemyHero", 124.35),
									L("Stealth"),
									L("AttackSafely"))),
							L("AttackHero")),
						N(D("DistanceToClosestEnemy", 120.0),
							N(D("DistanceToFront", -103.759),
								L("AttackSafely"),
								N(C("EnemySpells", ActionType.Stealth, ActionType.Powerup, ActionType.Bash),
									L("AttackHero"),
									N(C("EnemySpells", ActionType.Stealth, ActionType.ExplosiveShield, ActionType.AoeHeal, ActionType.Burning, ActionType.Bash),
										N(D("DistanceToClosestEnemyHero", 105.519),
											L("AttackHero"),
											L("Wire")),
										L("Wire")))),
							N(D("DistanceToFront", -40.0),
								L("Stealth"),
								N(D("NetHealthOfHeroesAdvantage", -20.0),
									L("AttackHero"),
									N(D("HealthOfClosestEnemyHero", 1450.0),
										L("AttackHero"),
										L("AttackSafely")))))),
					N(D("MyMana", 100.0),
						L("AttackSafely"),
						N(D("DistanceToMyTower", 899.208),
							N(D("DistanceToClosestEnemy", 311.0),
								N(D("HealthOfMyMinionsInFrontOfMe", 350.0),
									L("AttackSafely"),
									L("Stealth")),
								L("AttackSafely")),
							N(C("EnemySpells", ActionType.Stealth, ActionType.Counter, ActionType.AoeHeal, ActionType.Fireball, ActionType.Charge, ActionType.SpearFlip),
								L("Stealth"),
								N(D("DistanceToClosestEnemyHero", 90.602),
									N(C("EnemySpells", ActionType.Jump, ActionType.Pull),
										L("Wire"),
										L("AttackHero")),
									L("Wire")))))),
				N(C("EnemySpells", ActionType.Stealth, ActionType.Fireball, ActionType.Pull),
					N(D("TicksToEnemyHeroAttack", 1.961),
						N(C("EnemySpells", ActionType.Shield, ActionType.Counter, ActionType.Charge),
							L("AttackSafely"),
							N(C("EnemySpells", ActionType.Wire, ActionType.AoeHeal, ActionType.SpearFlip, ActionType.Bash),
								L("Stealth"),
								L("AttackSafely"))),
						N(D("DistanceToClosestAllyHero", 82.217),
							L("AttackHero"),
							L("Stealth"))),
					N(D("DistanceToMyTower", 1063.0),
						N(C("EnemySpells", ActionType.Powerup, ActionType.AoeHeal, ActionType.Bash, ActionType.Jump),
							N(C("EnemyHeroes", HeroType.DoctorStrange, HeroType.Hulk),
								N(D("TicksToEnemyHeroAttack", 0.499),
									L("Stealth"),
									L("AttackSafely")),
								L("Stealth")),
							L("AttackSafely")),
						N(D("NetHealthOfHeroesAdvantage", -85.0),
							L("Stealth"),
							N(D("HealthOfEnemyMinions", 1415.0),
								L("AttackHero"),
								L("Counter")))))),
			N(D("DistanceToClosestAllyHero", 160.015),
				N(C("EnemySpells", ActionType.Counter, ActionType.Burning, ActionType.Pull),
					N(D("HealthOfEnemyMinionsCloseToMe", 575.0),
						L("Counter"),
						N(C("EnemySpells", ActionType.ExplosiveShield, ActionType.Fireball),
							L("Stealth"),
							N(D("NetDamageToHeroesAdvantage", 55.0),
								L("Stealth"),
								L("Counter")))),
					N(C("EnemySpells", ActionType.Wire, ActionType.Powerup, ActionType.Pull),
						N(D("MaximumPotentialDamageToMe", 355.0),
							L("AttackHero"),
							L("Wire")),
						N(D("DistanceToClosestAllyHero", 135.0),
							L("Wire"),
							L("AttackSafely")))),
				N(D("MyHealth", 260.0),
					N(D("HealthOfClosestEnemyHero", 400.0),
						L("AttackSafely"),
						L("AttackHero")),
					N(D("MaximumPotentialEnemyHeroDamageToMe", 140.0),
						N(D("MaximumPotentialDamageToMe", 195.0),
							N(C("EnemyHeroes", HeroType.Deadpool, HeroType.DoctorStrange, HeroType.Ironman),
								L("Counter"),
								L("AttackHero")),
							L("Counter")),
						N(D("MyHealth", 622.0),
							L("Counter"),
							L("AttackSafely")))))),
		N(C("EnemyHeroes", HeroType.Ironman),
			N(D("MyMana", 100.0),
				L("AttackSafely"),
				N(D("DistanceToClosestEnemy", 291.0),
					N(D("DistanceToClosestEnemyHero", 201.191),
						L("Wire"),
						N(D("HealthOfEnemyMinionsCloseToMe", 1280.0),
							L("Stealth"),
							L("AttackHero"))),
					N(D("DistanceToWeakestOrClosestEnemyHero", 1067.6),
						N(D("DistanceToClosestEnemy", 341.0),
							N(D("DistanceToFront", 760.961),
								L("AttackSafely"),
								L("Stealth")),
							L("AttackSafely")),
						N(D("PotentialAggroDamageToWeakestOrClosestEnemy", 75.0),
							N(D("DistanceToClosestEnemy", 434.028),
								N(D("HealthOfEnemyMinions", 875.0),
									L("Retreat"),
									N(D("DistanceToFront", 811.0),
										L("AttackSafely"),
										L("Stealth"))),
								N(D("DistanceToClosestEnemy", 1061.0),
									L("AttackSafely"),
									N(D("DistanceToFront", 718.902),
										N(D("TicksToMinionSpawn", 12.0),
											L("Stealth"),
											N(D("DistanceToClosestEnemy", 1361.0),
												L("AttackSafely"),
												L("Stealth"))),
										L("AttackSafely")))),
							N(D("DistanceToClosestEnemy", 461.0),
								L("Stealth"),
								L("Retreat")))))),
			N(D("DistanceToClosestEnemy", 361.0),
				N(D("MyMana", 100.0),
					L("AttackSafely"),
					N(D("DistanceToClosestEnemy", 270.26),
						N(C("EnemySpells", ActionType.Wire, ActionType.AoeHeal, ActionType.Charge, ActionType.SpearFlip),
							L("Stealth"),
							L("Wire")),
						N(D("HealthOfEnemyMinions", 1650.0),
							N(D("HealthOfMyMinions", 644.0),
								L("AttackSafely"),
								L("Stealth")),
							L("AttackSafely")))),
				N(C("EnemyHeroes", HeroType.Valkyrie),
					L("Retreat"),
					N(C("AllySpells", ActionType.ExplosiveShield, ActionType.Jump),
						L("Retreat"),
						N(D("MyMana", 32.0),
							L("AttackSafely"),
							L("Stealth"))))))),
	DoctorStrange = N(D("DistanceToClosestEnemy", 460.89),
		N(D("DistanceToClosestEnemyHero", 400.87),
			N(D("HealthOfMyMinionsInFrontOfMe", 168.0),
				L("AttackSafely"),
				N(D("DistanceToMyTower", 359.0),
					N(D("HealthOfEnemyMinionsCloseToMe", 115.0),
						L("Pull"),
						L("AttackSafely")),
					L("AttackSafely"))),
			N(D("MyHealth", 921.0),
				N(D("DistanceToClosestAllyHero", 41.977),
					N(D("HealthOfMyMinionsCloseToMe", 20.0),
						N(C("MySpells", ActionType.AoeHeal),
							L("AttackSafely"),
							L("AoeHeal")),
						N(C("EnemySpells", ActionType.Shield, ActionType.Counter),
							L("AttackSafely"),
							N(D("HealthOfMyMinionsCloseToMe", 1390.0),
								L("AoeHeal"),
								L("AttackSafely")))),
					L("AttackSafely")),
				L("AttackSafely"))),
		N(D("HealthOfMyMinionsCloseToMe", 235.0),
			N(C("MySpells", ActionType.AoeHeal),
				N(D("DistanceToMyTower", 95.812),
					N(D("MyHealth", 669.0),
						L("Retreat"),
						N(D("DistanceToClosestAllyHero", 5.603),
							L("Retreat"),
							L("AttackSafely"))),
					L("AttackSafely")),
				L("AoeHeal")),
			N(D("DistanceToFront", -115.586),
				N(D("TicksToEnemyHeroAttack", 2.891),
					L("AttackSafely"),
					N(D("DistanceToClosestAllyNonHero", 140.339),
						L("Retreat"),
						L("AttackHero"))),
				L("Retreat")))),
	Hulk = N(D("DistanceToClosestEnemyHero", 295.857),
		N(D("MyMana", 30.0),
			L("AttackSafely"),
			N(D("NetDamageToHeroesAdvantage", -80.0),
				N(D("HealthOfClosestOrWeakestEnemyHero", 175.0),
					N(D("HealthOfMyMinions", 1545.0),
						L("ExplosiveShield"),
						L("Charge")),
					N(D("MaximumPotentialDamageToMe", 355.0),
						N(D("HealthOfMyMinionsInFrontOfMe", 400.0),
							N(D("DistanceToClosestAllyHero", 374.939),
								N(D("HealthOfMyMinionsCloseToMe", 540.0),
									N(D("DistanceToClosestEnemy", 95.0),
										N(C("EnemySpells", ActionType.Blink, ActionType.Shield, ActionType.Counter, ActionType.Powerup, ActionType.Fireball, ActionType.Charge, ActionType.SpearFlip, ActionType.Bash, ActionType.Jump, ActionType.Pull),
											L("ExplosiveShield"),
											N(D("MaximumPotentialDamageToMe", 220.0),
												N(D("NetDamageToHeroesAdvantage", -130.0),
													L("ExplosiveShield"),
													L("AttackHero")),
												L("ExplosiveShield"))),
										L("ExplosiveShield")),
									N(C("EnemySpells", ActionType.Shield, ActionType.SpearFlip),
										N(D("NetHealthOfMinionsAdvantage", -90.0),
											L("Charge"),
											L("AttackSafely")),
										N(D("MyHealth", 1150.0),
											L("Charge"),
											L("ExplosiveShield")))),
								N(C("EnemySpells", ActionType.Counter, ActionType.AoeHeal, ActionType.SpearFlip),
									N(D("NetHealthOfHeroesAdvantage", 70.0),
										N(D("TicksToDieFromStrongestInRangeHero", 11.0),
											L("ExplosiveShield"),
											L("Charge")),
										L("AttackSafely")),
									N(D("HealthOfClosestEnemyHero", 312.0),
										L("ExplosiveShield"),
										N(D("TicksToDieFromStrongestInRangeHero", 8.0),
											L("ExplosiveShield"),
											N(D("MyHealth", 1070.0),
												L("AttackSafely"),
												L("ExplosiveShield")))))),
							N(C("AllySpells", ActionType.Stealth, ActionType.Shield, ActionType.Fireball),
								L("ExplosiveShield"),
								L("AttackSafely"))),
						N(D("NetHealthOfMinionsAdvantage", 325.0),
							N(D("NetDamageToHeroesAdvantage", -365.0),
								L("ExplosiveShield"),
								N(C("EnemySpells", ActionType.Jump),
									L("ExplosiveShield"),
									N(D("NetHealthOfHeroesAdvantage", 50.0),
										N(D("HealthOfMyMinionsInFrontOfMe", 113.0),
											N(C("MyHeroes", HeroType.DoctorStrange, HeroType.Ironman, HeroType.Valkyrie),
												L("ExplosiveShield"),
												L("AttackSafely")),
											L("ExplosiveShield")),
										L("ExplosiveShield")))),
							L("ExplosiveShield")))),
				N(D("DistanceToMyTower", 966.449),
					N(D("DistanceToClosestEnemyHero", 130.0),
						N(C("MyHeroes", HeroType.Deadpool, HeroType.DoctorStrange, HeroType.Ironman),
							N(D("DistanceToWeakestOrClosestEnemyHero", 92.638),
								L("Charge"),
								N(C("EnemySpells", ActionType.Powerup, ActionType.Jump),
									L("Charge"),
									L("AttackHero"))),
							N(D("HealthOfClosestEnemyHero", 1400.0),
								N(D("NetDamageToHeroesAdvantage", 60.0),
									N(D("MaximumPotentialDamageToMe", 215.0),
										N(C("EnemySpells", ActionType.Blink, ActionType.Wire, ActionType.Powerup, ActionType.AoeHeal, ActionType.Fireball, ActionType.SpearFlip, ActionType.Jump),
											N(D("DistanceToClosestAllyNonHero", 48.252),
												L("Charge"),
												L("ExplosiveShield")),
											L("ExplosiveShield")),
										N(D("NetHealthOfHeroesAdvantage", 518.0),
											L("Charge"),
											L("ExplosiveShield"))),
									L("AttackSafely")),
								N(D("TicksToEnemyHeroAttack", 0.0),
									L("Bash"),
									L("AttackHero")))),
						L("Charge")),
					N(D("DistanceToClosestEnemyHero", 156.085),
						N(C("AllySpells", ActionType.Powerup),
							N(D("TicksToDieFromStrongestInRangeHero", 21.0),
								N(C("EnemyHeroes", HeroType.Deadpool, HeroType.DoctorStrange, HeroType.Valkyrie),
									L("Bash"),
									N(D("NetDamageToHeroesAdvantage", 25.0),
										N(D("DistanceToClosestAllyHero", 154.197),
											N(D("TicksToMinionSpawn", 6.0),
												N(D("NetHealthOfHeroesAdvantage", 200.0),
													L("ExplosiveShield"),
													L("Charge")),
												L("Bash")),
											N(D("DistanceToFront", -50.004),
												L("Charge"),
												N(D("PotentialAggroDamageToWeakestOrClosestEnemy", 50.0),
													L("ExplosiveShield"),
													L("AttackHero")))),
										N(D("HealthOfEnemyMinionsCloseToMe", 225.0),
											L("AttackHero"),
											L("ExplosiveShield")))),
								L("Charge")),
							N(D("HealthOfClosestOrWeakestEnemyHero", 1245.0),
								L("Bash"),
								L("ExplosiveShield"))),
						N(D("HealthOfMyMinionsCloseToMe", 780.0),
							L("AttackHero"),
							L("Charge")))))),
		N(D("MyMana", 30.0),
			N(D("DistanceToClosestEnemy", 481.0),
				L("AttackSafely"),
				N(D("DistanceToClosestAllyHero", 193.558),
					L("Retreat"),
					L("AttackSafely"))),
			N(D("DistanceToFront", 0.954),
				N(C("EnemySpells", ActionType.Stealth, ActionType.ExplosiveShield, ActionType.Fireball, ActionType.Pull),
					N(D("NetHealthOfHeroesAdvantage", -30.0),
						L("ExplosiveShield"),
						N(D("NetHealthOfHeroesAdvantage", 608.0),
							N(D("DistanceToMyTower", 900.014),
								N(C("AllySpells", ActionType.AoeHeal),
									N(D("DistanceToMyTower", 699.506),
										L("ExplosiveShield"),
										N(C("MyHeroes", HeroType.Ironman),
											L("AttackSafely"),
											L("AttackHero"))),
									L("ExplosiveShield")),
								L("ExplosiveShield")),
							N(D("NetHealthOfMinionsAdvantage", 0.0),
								L("AttackHero"),
								L("ExplosiveShield")))),
					N(D("DistanceToMyTower", 1099.653),
						N(D("TicksToEnemyHeroAttack", 6.375),
							N(D("DistanceToClosestEnemyHero", 1124.455),
								N(D("NetHealthOfHeroesAdvantage", 500.0),
									N(C("EnemyHeroes", HeroType.Valkyrie),
										N(D("HealthOfEnemyMinionsCloseToMe", 1200.0),
											L("AttackSafely"),
											L("AttackHero")),
										N(D("DistanceToFrontHero", 1.171),
											L("AttackHero"),
											L("AttackSafely"))),
									L("AttackHero")),
								N(C("MyHeroes", HeroType.DoctorStrange),
									L("AttackSafely"),
									L("AttackHero"))),
							N(D("HealthOfClosestEnemyHero", 1450.0),
								L("AttackHero"),
								L("ExplosiveShield"))),
						L("ExplosiveShield"))),
				N(D("HealthOfMyMinions", 685.0),
					N(D("DistanceToClosestEnemy", 209.5),
						L("AttackHero"),
						N(D("HealthOfMyMinions", 400.0),
							N(D("TicksToMinionSpawn", 2.0),
								L("Retreat"),
								L("ExplosiveShield")),
							N(D("HealthOfClosestEnemyHero", 1349.0),
								N(D("TicksToMinionSpawn", 5.0),
									N(D("NetHealthOfMinionsAdvantage", -475.0),
										L("ExplosiveShield"),
										N(D("HealthOfEnemyMinions", 585.0),
											L("Retreat"),
											N(C("EnemySpells", ActionType.Stealth, ActionType.Counter, ActionType.Powerup),
												L("AttackSafely"),
												N(D("NetHealthOfMinionsAdvantage", -315.0),
													L("AttackSafely"),
													L("Retreat"))))),
									L("AttackSafely")),
								N(D("DistanceToClosestEnemy", 358.441),
									L("Retreat"),
									L("AttackSafely"))))),
					N(D("HealthOfMyMinions", 1109.0),
						N(D("HealthOfClosestEnemyHero", 1375.0),
							N(D("DistanceToClosestEnemy", 259.793),
								L("AttackHero"),
								N(D("TicksToMinionSpawn", 9.0),
									N(D("NetHealthOfMinionsAdvantage", 20.0),
										N(D("DistanceToClosestEnemy", 562.0),
											L("AttackHero"),
											L("AttackSafely")),
										L("AttackHero")),
									L("ExplosiveShield"))),
							L("AttackHero")),
						N(D("HealthOfEnemyMinionsCloseToMe", 50.0),
							N(D("DistanceToClosestEnemy", 766.742),
								N(D("HealthOfClosestEnemyHero", 1380.0),
									N(C("AllySpells", ActionType.AoeHeal, ActionType.Jump),
										N(D("NetHealthOfMinionsAdvantage", -467.0),
											L("AttackHero"),
											N(D("DistanceToWeakestOrClosestEnemyHero", 471.298),
												L("ExplosiveShield"),
												N(D("DistanceToClosestEnemyHero", 1195.173),
													N(D("DistanceToWeakestOrClosestEnemyHero", 1122.567),
														N(C("MyHeroes", HeroType.Ironman, HeroType.Valkyrie),
															L("AttackHero"),
															L("ExplosiveShield")),
														L("Retreat")),
													L("AttackHero")))),
										L("AttackHero")),
									L("AttackSafely")),
								L("ExplosiveShield")),
							N(D("DistanceToClosestEnemy", 111.0),
								L("ExplosiveShield"),
								N(D("TicksToEnemyHeroAttack", 7.305),
									N(D("PotentialAggroDamageToMe", 70.0),
										L("AttackHero"),
										L("ExplosiveShield")),
									L("AttackHero"))))))))),
	Ironman = N(D("DistanceToClosestEnemyHero", 921.0),
		N(C("MySpells", ActionType.Fireball),
			N(C("MySpells", ActionType.Blink, ActionType.Burning),
				N(D("HealthOfEnemyMinions", 457.0),
					N(D("MaximumPotentialDamageToMe", 25.0),
						L("AttackHero"),
						L("Retreat")),
					N(D("DistanceToClosestEnemyHero", 420.556),
						N(D("HealthOfEnemyMinions", 1305.0),
							N(D("DistanceToClosestEnemy", 100.025),
								L("AttackSafely"),
								L("AttackHero")),
							L("AttackSafely")),
						L("AttackSafely"))),
				N(C("MySpells", ActionType.Blink),
					N(D("MyMana", 58.0),
						N(D("MaximumPotentialDamageToMe", 25.0),
							L("AttackSafely"),
							L("Retreat")),
						L("Retreat")),
					N(D("DistanceToClosestEnemy", 470.0),
						L("Retreat"),
						N(D("MyMana", 74.0),
							L("Blink"),
							L("Retreat"))))),
			L("Fireball")),
		N(D("DistanceToFront", 77.006),
			N(C("MySpells", ActionType.Blink),
				N(D("DistanceToClosestEnemy", 410.655),
					L("AttackHero"),
					L("Retreat")),
				L("AttackHero")),
			N(C("MySpells", ActionType.Blink),
				N(D("DistanceToClosestEnemy", 318.919),
					N(D("DistanceToWeakestOrClosestEnemyHero", 1341.702),
						N(C("MySpells", ActionType.Fireball),
							N(D("MyMana", 58.0),
								N(D("HealthOfEnemyMinions", 455.0),
									L("AttackHero"),
									L("AttackSafely")),
								L("Retreat")),
							L("AttackHero")),
						N(D("DistanceToFront", 802.534),
							L("AttackHero"),
							L("AttackSafely"))),
					N(D("DistanceToFront", 761.0),
						N(D("MyMana", 58.0),
							L("AttackSafely"),
							L("Retreat")),
						N(D("DistanceToClosestEnemyHero", 1322.0),
							N(D("DistanceToClosestEnemyHero", 1121.0),
								N(C("MySpells", ActionType.Fireball),
									N(D("MyMana", 50.0),
										L("AttackSafely"),
										L("Retreat")),
									L("AttackSafely")),
								N(D("MyMana", 50.0),
									L("AttackSafely"),
									L("Retreat"))),
							N(C("MySpells", ActionType.Fireball),
								N(D("MyMana", 50.0),
									L("AttackSafely"),
									L("Retreat")),
								L("AttackSafely"))))),
				L("Retreat")))),
	Valkyrie = N(D("DistanceToMyTower", 107.426),
		N(D("HealthOfMyMinionsInFrontOfMe", 615.0),
			L("Retreat"),
			N(D("DistanceToFront", 371.0),
				N(D("HealthOfMyMinions", 1465.0),
					L("Retreat"),
					L("AttackSafely")),
				N(D("NetHealthOfMinionsAdvantage", -5.0),
					L("AttackSafely"),
					N(C("EnemyHeroes", HeroType.Ironman),
						L("AttackSafely"),
						N(D("HealthOfMyMinions", 1455.0),
							N(C("EnemySpells", ActionType.Stealth, ActionType.ExplosiveShield, ActionType.Fireball),
								L("Retreat"),
								L("AttackSafely")),
							L("AttackSafely")))))),
		N(D("DistanceToClosestEnemyHero", 160.042),
			N(D("HealthOfEnemyMinionsCloseToMe", 650.0),
				L("SpearFlip"),
				L("AttackSafely")),
			N(D("DistanceToClosestEnemy", 68.966),
				N(D("DistanceToMyTower", 781.481),
					N(C("MySpells", ActionType.Powerup),
						L("AttackSafely"),
						N(D("DistanceToClosestAllyNonHero", 240.353),
							N(D("PotentialAggroDamageToWeakestOrClosestEnemy", 60.0),
								L("AttackSafely"),
								L("AttackHero")),
							L("AttackHero"))),
					N(C("MySpells", ActionType.Powerup),
						L("AttackSafely"),
						L("Retreat"))),
				N(D("DistanceToClosestEnemy", 861.395),
					N(D("HealthOfMyMinionsCloseToMe", 805.0),
						L("AttackSafely"),
						N(C("MySpells", ActionType.Powerup),
							L("AttackSafely"),
							N(D("HealthOfMyMinionsCloseToMe", 1275.0),
								L("Powerup"),
								N(D("TicksToMinionSpawn", 13.0),
									N(D("DistanceToClosestEnemy", 501.768),
										N(D("HealthOfMyMinionsInFrontOfMe", 287.0),
											L("AttackSafely"),
											N(D("DistanceToClosestEnemy", 161.0),
												L("AttackSafely"),
												N(D("TicksToMinionSpawn", 11.0),
													L("Powerup"),
													L("AttackSafely")))),
										N(D("NetHealthOfMinionsAdvantage", -87.0),
											L("AttackSafely"),
											N(D("HealthOfMyMinions", 1505.0),
												L("Powerup"),
												L("AttackSafely")))),
									L("AttackSafely"))))),
					N(C("EnemyHeroes", HeroType.Hulk, HeroType.Ironman),
						N(D("MyMana", 135.0),
							L("AttackSafely"),
							L("AttackHero")),
						L("AttackSafely")))))),
};



	}
}
