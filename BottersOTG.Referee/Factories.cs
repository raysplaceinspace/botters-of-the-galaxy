using System;
using System.Collections.Generic;

namespace BOTG_Refree
{
	public class Factories
	{
		enum HeroType
		{
			IRONMAN,
			DEADPOOL,
			DOCTOR_STRANGE,
			VALKYRIE,
			HULK
		}

		public static LaneUnit generateUnit(int type, int team, int number, Player player)
		{
			Point spawn = team == 0 ? Const.SPAWNTEAM0 : Const.SPAWNTEAM1;
			Point target = team == 1 ? Const.TOWERTEAM0 : Const.TOWERTEAM1;
			LaneUnit unit = new LaneUnit(spawn.x + 50 * type * (team * 2 - 1), spawn.y + number * 50, 1, team, 150, target, player);
			unit.targetPoint = new Point(unit.targetPoint.x, unit.y); // Move in a straight line
																	  

			if (type == 0)
			{
				unit.health = unit.maxHealth = 400;
				unit.damage = 25;
				unit.range = 90;
				unit.goldValue = Const.MELEE_UNIT_GOLD_VALUE;
			}
			else
			{
				unit.health = unit.maxHealth = 250;
				unit.damage = 35;
				unit.range = 300;
				unit.goldValue = Const.RANGER_UNIT_GOLD_VALUE;
			}
			unit.attackTime = 0.2;

			return unit;
		}

		public static Hero generateHero(string type, Player player, Point spawn)
		{
			int team = player.getIndex();
			Hero hero = new Hero(spawn.x, spawn.y, 1, team, 200, player, type);

			// Since stub doesn't support IFs, just take heroes
			if (type.StartsWith("WAIT"))
			{
				if (Const.game.round == 0) type = HeroType.HULK.ToString();
				else if (Const.game.round == 1 && player.heroes[0].heroType == "IRONMAN") type = HeroType.DEADPOOL.ToString();
				else if (Const.game.round == 1) type = HeroType.IRONMAN.ToString();
				hero.heroType = type;
			}

			if (type == HeroType.IRONMAN.ToString())
			{
				hero.health = hero.maxHealth = 820;
				hero.mana = hero.maxMana = 200;
				hero.damage = 60;
				hero.range = 270;
				hero.manaregeneration = 2;
				hero.skills[0] = new Skills.BlinkSkill(hero, 16, "BLINK", 0.05, true, 200, 3);
				hero.skills[1] = new Skills.LineSkill(hero, 60, "FIREBALL", 900, 50, 0.9, 6);
				hero.skills[2] = new Skills.BurningGround(hero, 50, "BURNING", 0.01, 250, 100, "ENERGY-BALL.png", 5);

			}
			else if (type == HeroType.VALKYRIE.ToString())
			{
				hero.health = hero.maxHealth = 1400;
				hero.mana = hero.maxMana = 155;
				hero.damage = 65;
				hero.range = 130;
				hero.manaregeneration = 2;
				hero.skills[0] = new Skills.FlipSkill(hero, 20, "SPEARFLIP", 0.1, 155, 3);
				hero.skills[1] = new Skills.JumpSkill(hero, 35, "JUMP", 0.15, 250, 3);
				hero.skills[2] = new Skills.PowerUpSkill(hero, 50, "POWERUP", 4, 0, 7);

			}
			else if (type == HeroType.DEADPOOL.ToString())
			{
				hero.health = hero.maxHealth = 1380;
				hero.mana = hero.maxMana = 100;
				hero.damage = 80;
				hero.range = 110;
				hero.manaregeneration = 1;
				hero.skills[0] = new Skills.CounterSkill(hero, 40, "COUNTER", 1, 350, 5);
				hero.skills[1] = new Skills.WireHookSkill(hero, 50, "WIRE", 200, 25, 2, 0.3, 9);
				hero.skills[2] = new Skills.StealthSkill(hero, 30, "STEALTH", 0, 5, 6);

			}
			else if (type == HeroType.DOCTOR_STRANGE.ToString())
			{
				hero.health = hero.maxHealth = 955;
				hero.mana = hero.maxMana = 300;
				hero.damage = 50;
				hero.range = 245;
				hero.manaregeneration = 2;
				hero.skills[0] = new Skills.AOEHealSkill(hero, 50, "AOEHEAL", 0.01, 250, 100, "HEALING-BALL.png", 6);
				hero.skills[1] = new Skills.ShieldSkill(hero, 40, "SHIELD", 3, 500, 6);
				hero.skills[2] = new Skills.PullSkill(hero, 40, "PULL", 0.3, 400, 5, 0.1);

			}
			else if (type == HeroType.HULK.ToString())
			{
				hero.health = hero.maxHealth = 1450;
				hero.mana = hero.maxMana = 90;
				hero.damage = 80;
				hero.range = 95;
				hero.manaregeneration = 1;
				hero.skills[0] = new Skills.ChargeSkill(hero, 20, "CHARGE", 0.05, 300, 4);
				hero.skills[1] = new Skills.ExplosiveSkill(hero, 30, "EXPLOSIVESHIELD", 4, 100, 8);
				hero.skills[2] = new Skills.BashSkill(hero, 40, "BASH", 2, 150, 10);

			}
			else
			{
				throw new InvalidInputException("Hero not supported", type);
			}

			if (Const.IGNORESKILLS)
			{
				hero.skills[0] = new Skills.EmptySkill();
				hero.skills[1] = new Skills.EmptySkill();
				hero.skills[2] = new Skills.EmptySkill();
			}

			return hero;
		}

		public static Tower generateTower(Player player, int team)
		{
			Point spawn = team == 0 ? Const.TOWERTEAM0 : Const.TOWERTEAM1;
			Tower tower = new Tower(spawn.x, spawn.y, (int)(Const.TOWERHEALTH * Const.TOWERHEALTHSCALE), team, player);
			tower.range = 400;
			tower.damage = Const.TOWERDAMAGE;
			player.tower = tower;
			tower.attackTime = 0.2;

			return tower;
		}

		public static Bush generateBush(Point point)
		{
			return new Bush(point.x, point.y);
		}

		public static Creature generateCreature(Point point, double amplitude)
		{
			Creature creature = new Creature(point.x, point.y);
			//creature.skin = Const.GROOT;
			creature.health = creature.maxHealth = (int)(400 * amplitude);
			creature.damage = (int)(35 * amplitude);
			creature.range = 150;
			creature.moveSpeed = 250;
			creature.goldValue = (int)(Const.NEUTRALGOLD * amplitude);
			creature.creatureType = "GROOT";
			creature.attackTime = 0.2;

			return creature;
		}

		public static Dictionary<string, Item> createItems(int playersGold)
		{
			if (Const.IGNOREITEMS)
				return new Dictionary<string, Item>();
			return MapFactory.generateItems(playersGold);
		}
	}
}