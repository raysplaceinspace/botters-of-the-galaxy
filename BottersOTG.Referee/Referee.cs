using System;
using System.Collections.Generic;
using System.Linq;

namespace BOTG_Refree
{
	public class Referee
	{
		public GameManager<Player> gameManager;
		private const int LostScore = -100000000;

		public static void setupLeague(int league)
		{
			switch (league)
			{
				case 0:
					Const.IGNOREITEMS = true;
					Const.HEROCOUNT = 1;
					Const.IGNORESKILLS = true;
					Const.REMOVEFORESTCREATURES = true;
					Const.TOWERHEALTHSCALE = 0.5;
					Const.IGNOREBUSHES = true;
					Const.Rounds = 200;
					Const.TOWERDAMAGE = 1;
					Const.MELEE_UNIT_COUNT = 0;
					Const.RANGED_UNIT_COUNT = 0;
					break;
				case 1:
					Const.HEROCOUNT = 1;
					Const.IGNORESKILLS = true;
					Const.REMOVEFORESTCREATURES = true;
					Const.TOWERHEALTHSCALE = 0.5;
					Const.IGNOREBUSHES = true;
					Const.Rounds = 200;
					Const.TOWERDAMAGE = 1;
					Const.MELEE_UNIT_COUNT = 0;
					Const.RANGED_UNIT_COUNT = 0;
					break;
				case 2:
					Const.HEROCOUNT = 1;
					Const.IGNORESKILLS = true;
					Const.REMOVEFORESTCREATURES = true;
					Const.TOWERHEALTHSCALE = 0.5;
					Const.IGNOREBUSHES = true;
					Const.Rounds = 200;
					Const.TOWERDAMAGE = 1;
					break;
				case 3:
					Const.HEROCOUNT = 1;
					Const.IGNORESKILLS = true;
					Const.REMOVEFORESTCREATURES = true;
					Const.TOWERHEALTHSCALE = 0.5;
					Const.Rounds = 200;
					break;
				case 4:
					Const.HEROCOUNT = 1;
					Const.IGNORESKILLS = true;
					Const.Rounds = 200;
					break;
				case 5:
					Const.IGNORESKILLS = true;
					Const.Rounds = 200;
					break;
				default:
					//normal.
					break;
			}
		}


		public Properties init(Properties _params)
		{
			int seed = 42;
			try
			{
				seed = int.Parse(_params.getProperty("seed", ((int)DateTime.Now.Ticks).ToString()));
			}
			catch (Exception /*e*/)
			{
				seed = new Random().Next();
			}
			_params.Add("seed", seed.ToString());
			Const.random = new Random(seed);

			setupLeague(gameManager.getLeagueLevel() - 1);

			gameManager.setMaxTurns(Const.Rounds);
			Const.game.initialize(gameManager.getActivePlayers());

			foreach (Unit unit in Const.game.allUnits)
			{
				if (unit is Tower)
					gameManager.getActivePlayers()[unit.team].tower = (Tower)unit;
			}

			return _params;
		}

		public void gameTurn(int turn)
		{
			Const.game.beforeTurn(turn, gameManager.getActivePlayers());

			if (turn == 0) sendInitialData();

			foreach (Player player in gameManager.getPlayers())
			{
				player.sendInputLine($"{player.getGold()}");
				player.sendInputLine($"{getOther(player).getGold()}");

				int hc = (turn >= Const.HEROCOUNT) ? player.heroes.Where(h => !h.isDead).Count() : (turn - Const.HEROCOUNT);

				player.sendInputLine($"{hc}");
				var sendingUnits = Const.game.allUnits.Where(u => shouldSendToPlayer(player, u));
				player.sendInputLine($"{sendingUnits.Count()}");

				foreach (Unit unit in sendingUnits)
				{
					player.sendInputLine(unit.getPlayerString());
				}
			}

			foreach (Player player in gameManager.getPlayers())
			{
				player.execute();
				string[] strinOutputs = new string[0];
				if (turn < Const.HEROCOUNT)
				{
					gameManager.setFrameDuration(100);
					pickHero(player);
				}
				else
				{
					gameManager.setFrameDuration(1000);
					try
					{
						string[] outputs = player.getOutputs(player.heroesAlive());
						player.handlePlayerOutputs(outputs);
					}
					catch (Exception e)
					{
						player.setScore(LostScore);
						string msg;
						if (e is TimeoutException)
							msg = " timeout";
						else
							msg = " " + e.Message;
						player.deactivate(player.getNicknameToken() + msg);
					}
				}
			}

			if (gameManager.getActivePlayers().Count == 2)
				Const.game.handleTurn(gameManager.getActivePlayers());



			foreach (Player player in gameManager.getActivePlayers())
			{
				player.setScore(0);
				bool deadHeroes = player.heroes.All(h => h.isDead);
				if (deadHeroes || player.tower.isDead)
				{
					player.setScore(LostScore);
					player.deactivate(player.getNicknameToken() + " lost. " + (deadHeroes ? "All heroes dead" : "Tower destroyed"));
				}
				else
				{
					player.setScore(player.unitKills + player.denies);
				}
			}

			if (gameManager.getActivePlayers().Count < 2)
			{
				gameManager.endGame();
			}
		}

		private bool shouldSendToPlayer(Player player, Unit u)
		{
			bool r = !u.isDead && (u.team == player.getIndex() || u.visible || u.becomingInvis || !(u is Hero));
			return !u.isDead && (u.team == player.getIndex() || u.visible || u.becomingInvis || !(u is Hero));
		}

		private Player getOther(Player player)
		{
			return gameManager.getActivePlayers()[1 - player.getIndex()];
		}

		private void pickHero(Player player)
		{
			string output = "";
			try
			{
				Point spawn = player.getIndex() == 0 ? (player.heroes.Count == 0 ? Const.HEROSPAWNTEAM0 : Const.HEROSPAWNTEAM0HERO2) : (player.heroes.Count == 0 ? Const.HEROSPAWNTEAM1 : Const.HEROSPAWNTEAM1HERO2);
				output = player.getOutputs(1)[0];
				if (player.heroes.Count > 0 && player.heroes[0].heroType == output)
				{
					player.setScore(LostScore);
					player.deactivate(player.getNicknameToken() + " tried to pick a hero already owned. Can't have duplicate hero.");
					gameManager.addToGameSummary(player.getNicknameToken() + " tried to pick a hero already owned. Can't have duplicate hero.");
					return;
				}
				else
				{
					Hero hero = Factories.generateHero(output, player, spawn);
					player.addHero(hero);
					Const.game.allUnits.Add(hero);
				}
			}
			catch (Exception e)
			{
				player.setScore(LostScore);
				string msg;
				if (e is TimeoutException)
					msg = " timeout";
				else
					msg = " supplied invalid input. " + e.Message;
				player.deactivate(player.getNicknameToken() + msg);
			}
		}

		private void sendInitialData()
		{
			foreach (Player player in gameManager.getActivePlayers())
			{
				player.sendInputLine($"{player.getIndex()}");

				player.sendInputLine($"{Const.game.bushes.Count + Const.game.spawns.Length}");
				foreach (Bush bush in Const.game.bushes)
				{
					player.sendInputLine(bush.getPlayerString());
				}

				foreach (CreatureSpawn spawn in Const.game.spawns)
				{
					player.sendInputLine(spawn.getPlayerString());
				}

				//ITEMS
				player.sendInputLine($"{Const.game.items.Count}");
				foreach (string itemName in Const.game.items.Keys)
				{
					Item item = Const.game.items[itemName];
					player.sendInputLine(item.getPlayerString());
				}
			}
		}
	}
}
