using System;
using System.Collections.Generic;

namespace BOTG_Refree
{

	public static class MapFactory
	{
		private static int center = Const.MAPWIDTH / 2;

		public static List<Bush> generateBushes(CreatureSpawn[] spawns)
		{
			List<Bush> obstacles = new List<Bush>();
			// bushes below the lane
			int posX = 50;
			int bottomBushCount = (int)(Const.random.NextDouble() * 4 + 1);
			for (int i = 0; i < bottomBushCount; i++)
			{
				posX = (int)(Const.random.NextDouble() * ((910 - posX) / (bottomBushCount - i)) + posX);
				if (posX > 910) break;
				obstacles.Add(Factories.generateBush(new Point(posX, 720)));
				obstacles.Add(Factories.generateBush(new Point(1920 - posX, 720)));
				posX += 100;
			}

			// bushes above the lane
			posX = 50;
			bottomBushCount = (int)(Const.random.NextDouble() * 2 + 2);
			for (int i = 0; i < bottomBushCount; i++)
			{
				posX = (int)(Const.random.NextDouble() * ((910 - posX) / (bottomBushCount - i)) + posX);
				if (posX > 910) break;
				obstacles.Add(Factories.generateBush(new Point(posX, 380)));
				obstacles.Add(Factories.generateBush(new Point(1920 - posX, 380)));
				posX += 100;
			}

			int extraBushes = (spawns.Length);
			for (int i = 0; i < extraBushes; i++)
			{
				int posY = (int)spawns[i].y;
				posX = (int)spawns[i].x;
				spawnExtraBushes(posX - 150, posY - 150, posX + 150, posY + 150, obstacles);
			}

			extraBushes = (int)(Const.random.NextDouble() * 4);
			for (int i = 0; i < extraBushes; i++)
			{
				spawnExtraBushes(Const.MAPWIDTH / 2 - 300, 50, Const.MAPWIDTH / 2 + 300, 300, obstacles);
			}
			return obstacles;
		}


		static void spawnExtraBushes(int startX, int startY, int endX, int endY, List<Bush> bushes)
		{
			for (int i = 0; i < 10; i++)
			{
				int posX = (int)(Const.random.NextDouble() * (endX - startX) + startX);
				int posY = (int)(Const.random.NextDouble() * (endY - startY) + startY);
				if (posX < Const.BUSHRADIUS
						|| posY < Const.BUSHRADIUS
						|| posX > Const.MAPWIDTH - Const.BUSHRADIUS
						|| posY > Const.MAPHEIGHT - Const.BUSHRADIUS) continue;
				Point newBush = new Point(posX, posY);
				bool isOverlapping = false;
				foreach (Bush bush in bushes)
				{
					if (bush.Distance(newBush) <= bush.radius * 2) isOverlapping = true;
				}

				if (!isOverlapping)
				{
					if (Math.Abs(posX - center) < Const.BUSHRADIUS)
						bushes.Add(Factories.generateBush(new Point(center, posY)));
					else
					{
						bushes.Add(Factories.generateBush(new Point(posX, posY)));
						bushes.Add(Factories.generateBush(new Point(1920 - posX, posY)));
					}
					break;
				}
			}
		}


		internal static CreatureSpawn[] generateSpawns()
		{
			int forestCampCount = (int)(Const.random.NextDouble() * 2 + 3);
			CreatureSpawn[] spawns = new CreatureSpawn[forestCampCount];
			int posX = 50;
			int campIter = forestCampCount / 2;
			int midBoundary = 850;
			if (forestCampCount % 2 > 0)
			{
				spawns[forestCampCount / 2] = new CreatureSpawn(960, (int)(Const.random.NextDouble() * 230 + 50));
				midBoundary = 750;
			}
			for (int i = 0; i < campIter; i++)
			{
				posX = (int)(Const.random.NextDouble() * ((midBoundary - posX) / (campIter - i)) + posX);
				if (posX > midBoundary) break;
				int posY = (int)(Const.random.NextDouble() * 230 + 50);
				spawns[i] = new CreatureSpawn(posX, posY);
				spawns[forestCampCount - i - 1] = new CreatureSpawn(1920 - posX, posY);
				posX += 150;
			}

			return spawns;
		}

		// starting - early - mid - late

		public static int[,] ItemLevels = { { 50, 200 }, { 200, 550 }, { 550, 1300 }, { 1300, 3000 } };
		public static Dictionary<string, Item> generateItems(int playersGold)
		{
			Dictionary<string, Item> items = new Dictionary<string, Item>();

			for (int l = 0; l < ItemLevels.GetLength(0); l++)
			{
				for (int i = 0; i < Const.NB_ITEMS_PER_LEVEL; i++)
				{
					addItem(generateItem(ItemLevels[l, 0], ItemLevels[l, 1], items.Count + 1), items);
				}
			}
			addItem(generateManaPot(), items);
			addItem(generateLargePot(), items);
			addItem(generateXXLPot(), items);
			return items;
		}

		static void addItem(Item item, Dictionary<string, Item> items)
		{
			items[item.name] = item;
		}

		static Item generateItem(int lowerLimit, int upperLimit, int nb)
		{
			Dictionary<string, int> stats = new Dictionary<string, int>();

			int spent = 0;
			string prefix = "";
			string suffix = "Gadget";
			int cost = Utilities.rndInt(lowerLimit, upperLimit);

			while (spent + Const.MINIMUM_STAT_COST < cost)
			{

				int investment = Math.Min(cost - spent, Utilities.rndInt(Const.MINIMUM_STAT_COST, cost));
				string statType = Const.STATS[Utilities.rndInt(0, Const.STATS.Length)];

				if (stats.ContainsKey(statType))
				{
					if (stats[statType] < getLimit(statType))
					{
						int current = stats[statType];
						int extra = (int)(investment / getPrice(statType));
						int actual = Math.Min(getLimit(statType), current + extra);
						stats[statType] = actual;
						investment = (int)((actual - current) * getPrice(statType));
					}
					else continue;
				}
				else
				{
					int extra = (int)(investment / getPrice(statType));
					int actual = Math.Min(getLimit(statType), extra);
					stats[statType] = actual;
					investment = (int)((actual) * getPrice(statType));
				}

				spent += investment;
			}

			if (stats.ContainsKey(Const.HEALTH))
			{
				stats[Const.MAXHEALTH] = stats[Const.HEALTH];
			}
			if (stats.ContainsKey(Const.MANA))
			{
				stats[Const.MAXMANA] = stats[Const.MANA];
			}

			int dmg = stats.ContainsKey(Const.DAMAGE) ? stats[Const.DAMAGE] : 0;
			int speed = stats.ContainsKey(Const.MOVESPEED) ? stats[Const.MOVESPEED] : 0;

			if (dmg > 0 && dmg > speed)
			{
				suffix = "Blade";
			}
			else if (speed > 0)
			{
				suffix = "Boots";
			}
			if (lowerLimit == ItemLevels[0, 0]) prefix += "Bronze";
			else if (lowerLimit == ItemLevels[1, 0]) prefix += "Silver";
			else if (lowerLimit == ItemLevels[2, 0]) prefix += "Golden";
			else if (lowerLimit == ItemLevels[3, 0]) prefix += "Legendary";
			return new Item(prefix + "_" + suffix + "_" + nb, stats, findCost(stats), false);
		}

		static Item generateManaPot()
		{
			Dictionary<string, int> stats = new Dictionary<string, int>();
			stats[Const.MANA] = 50;
			Item pot = new Item("mana_potion", stats, findCost(stats), true);
			pot.isPotion = true;
			return pot;
		}

		static Item generateLargePot()
		{
			Dictionary<string, int> stats = new Dictionary<string, int>();
			stats[Const.HEALTH] = 100;
			Item pot = new Item("larger_potion", stats, findCost(stats), true);
			pot.isPotion = true;
			return pot;
		}

		static Item generateXXLPot()
		{
			Dictionary<string, int> stats = new Dictionary<string, int>();
			stats[Const.HEALTH] = 500;
			Item pot = new Item("xxl_potion", stats, findCost(stats), true);
			pot.isPotion = true;
			return pot;
		}

		public const double healthPrice = 0.7;
		public const double maxHealthPrice = 0;
		public const double manaPrice = 0.5;
		public const double maxManaPrice = 0;
		public const double damagePrice = 7.2;
		public const double moveSpeedPrice = 3.6;
		public const double manaRegPrice = 50;

		static int findCost(Dictionary<string, int> stats)
		{
			double totalCost = 0.0;
			foreach (string stat in stats.Keys)
			{
				int val = stats[stat];
				totalCost += val * getPrice(stat);
			}

			return (int)Math.Min(Math.Max(Math.Ceiling(totalCost / 2.5), Math.Ceiling(totalCost - (totalCost * totalCost / 5000))), 1200);
		}

		public static int getLimit(string type)
		{
			switch (type)
			{
				case Const.HEALTH: return 2500;
				case Const.MAXHEALTH: return 2500;
				case Const.MANA: return 100;
				case Const.MAXMANA: return 100;
				case Const.DAMAGE: return 100000000;
				case Const.MOVESPEED: return 150;
				case Const.MANAREGEN: return 50;
			}

			return 0;
		}

		public static double getPrice(string type)
		{
			switch (type)
			{
				case Const.HEALTH: return healthPrice;
				case Const.MAXHEALTH: return maxHealthPrice;
				case Const.MANA: return manaPrice;
				case Const.MAXMANA: return maxManaPrice;
				case Const.DAMAGE: return damagePrice;
				case Const.MOVESPEED: return moveSpeedPrice;
				case Const.MANAREGEN: return manaRegPrice;
			}

			return 0;
		}

		static double addItemCost(int val, string stat, string type, double amplitude)
		{
			if (stat == type) return val * amplitude;
			return 0.0;
		}
	}
}
