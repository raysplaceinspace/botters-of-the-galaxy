using System;
using System.Collections.Generic;
using System.Linq;

namespace BOTG_Refree
{
	public class Game
	{
		// following two lists will be sent individually to the players, based on what heroes are visible

		public List<Unit> allUnits = new List<Unit>();
		public List<Bush> bushes = new List<Bush>();
		public List<Event> events = new List<Event>();
		public Dictionary<string, Item> items;
		public int round;
		public List<Hero> visibleHeroes = new List<Hero>();
		public int forestCampCount = 0;
		public Dictionary<Unit, List<Damage>> damages = new Dictionary<Unit, List<Damage>>();
		public double t;

		public void handleTurn(List<Player> players)
		{

			// Make dummies do their actions
			foreach (Unit unit in allUnits)
			{
				if (unit.stunTime > 0)
					continue;
				unit.findAction(allUnits);
			}

			// Inform events about speed changes of moving units.
			foreach (Unit unit in allUnits)
			{
				if (Math.Abs(unit.vx) > Const.EPSILON || Math.Abs(unit.vy) > Const.EPSILON)
				{
					for (int i = events.Count - 1; i >= 0; i--)
					{
						Event _event = events[i];
						if (_event.afterAnotherEvent(unit, Event.SPEEDCHANGED, 0.0))
						{
							events.Remove(_event);
						}
					}
				}
			}

			while (t <= Const.ROUNDTIME)
			{
				if (isGameOver(players))
				{
					return;
				}

				Event nextEvent = findNextEvent();
				if (nextEvent == null || nextEvent.t + t > Const.ROUNDTIME)
				{
					break;
				}

				events.Remove(nextEvent);
				double eventTime = nextEvent.t + Const.EPSILON;

				foreach (Unit unit in allUnits)
				{
					unit.move(eventTime);
				}

				foreach (Event _event in events)
				{
					_event.t -= eventTime;
				}

				t += eventTime;

				List<Event> occuringEvents = new List<Event>();
				occuringEvents.Add(nextEvent);
				for (int i = events.Count - 1; i >= 0; i--)
				{
					Event _event = events[i];
					if (_event.t < 0 && _event.t + t <= 1.0)
					{
						occuringEvents.Add(_event);
						events.RemoveAt(i);
					}
				}

				occuringEvents.Sort();

				foreach (Event currentEvent in occuringEvents)
				{
					List<Unit> affectedUnits = currentEvent.onEventTime(t);

					handleAfterEvents(affectedUnits, currentEvent.getOutcome());
				}

				handleDamage();
			} // end while

			foreach (Event _event in events)
			{
				_event.t -= 1.0 - t;
			}

			foreach (Unit unit in allUnits)
			{
				unit.move(1.0 - t);
			}

			for (int i = events.Count - 1; i >= 0; i--)
			{
				Event _event = events[i];
				if (!_event.useAcrossRounds())
				{
					events.Remove(_event);
				}
			}

			foreach (Unit unit in allUnits)
			{
				unit.afterRound();
			}

			updateVisibility(players);
		}

		private void handleDamage()
		{
			foreach (var entry in damages)
			{
				Unit target = entry.Key;
				List<Damage> dmgs = entry.Value;
				int totalDamage = 0;
				bool anyHero = target is Hero;

				Unit highestDamageUnit = null;

				// Used when 2 meele attacks groot
				Unit otherAttacker = null;

				int highestDmg = 0;

				foreach (Damage dmg in dmgs)
				{
					totalDamage += dmg.damage;
					anyHero = anyHero || dmg.attacker is Hero;

					// hero becomes visible if he damages any on enemy team
					if (dmg.attacker is Hero && target.player != null && target.player != dmg.attacker.player)
					{
						((Hero)dmg.attacker).visibilityTimer = 2;
						dmg.attacker.visible = true;
						dmg.attacker.invisibleBySkill = false;
					}
					else if (dmg.attacker is Hero)
						dmg.attacker.invisibleBySkill = false;

					// Highest damage or attackers advantage.
					if (dmg.damage > highestDmg || (dmg.damage == highestDmg && dmg.attacker.team == 1 - dmg.target.team))
					{
						highestDamageUnit = dmg.attacker;
						highestDmg = dmg.damage;
						otherAttacker = null;
					}
					else if (dmg.damage == highestDmg && dmg.target.team == -1 && dmg.attacker.team != highestDamageUnit.team)
					{
						otherAttacker = dmg.attacker;
					}
				}
				if (totalDamage > 0 && target.getShield() > 0)
				{
					int val = Math.Min(totalDamage, target.getShield());
					target.adjustShield(val);
					totalDamage -= val;
				}

				target.health = Math.Min(target.health - totalDamage, target.maxHealth);

				if (target.health <= 0)
				{
					target.isDead = true;

					UnitKilledState state = highestDamageUnit is Hero ? UnitKilledState.farmed : UnitKilledState.normal;

					if (highestDamageUnit.team == target.team)
					{
						target.goldValue = 0;
						if (highestDamageUnit.player != null)
							highestDamageUnit.player.denies++;

						state = UnitKilledState.denied;
					}

					if (target is Hero)
					{
						// dead men tell no tales

						if (highestDamageUnit is LaneUnit || highestDamageUnit is Tower)
						{
							target.goldValue = target.goldValue / 2; // hero lost half of its value if killed by creep or tower
						}
						else if (highestDamageUnit is Creature)
						{
							target.goldValue = 0;
						}
					}
					else if (highestDamageUnit is Hero && highestDamageUnit.team != target.team)
					{
						if (highestDamageUnit.player != null)
							highestDamageUnit.player.unitKills++;
					}
					else
					{
						target.goldValue = 0;
					}

					if (otherAttacker != null && otherAttacker is Hero)
					{
						otherAttacker.player.unitKills++;
						otherAttacker.player.gold += target.goldValue;
					}

					if (highestDamageUnit.player != null)
						highestDamageUnit.player.gold += target.goldValue;
					System.Diagnostics.Debug.WriteLine($"{highestDamageUnit.player.player_id} killed {target.getType()}");
					Const.game.allUnits.Remove(target);
				}
				else if (target is Creature)
				{
					Creature c = (Creature)target;
					if (c.state == CreatureState.peacefull)
					{
						c.state = CreatureState.aggressive;
					}
				}

				List<Unit> affectedUnits = new List<Unit>();
				affectedUnits.Add(target);
				handleAfterEvents(affectedUnits, Event.LIFECHANGED);
			}

			damages.Clear();
		}

		private void handleAfterEvents(List<Unit> affectedUnits, int outcome)
		{
			for (int i = events.Count - 1; i >= 0; i--)
			{
				Event _event = events[i];
				foreach (Unit unit in affectedUnits)
				{
					if (_event.afterAnotherEvent(unit, outcome, t) || _event.t < 0)
					{
						events.Remove(_event);
					}
				}
			}
		}

		// updates visibility of heroes based on their positions in relation to bushes, trees and other heroes
		private void updateVisibility(List<Player> players)
		{
			visibleHeroes.Clear();
			foreach (Player player in players)
			{
				foreach (Hero hero in player.heroes)
				{
					if (hero.isDead) continue;
					visibleHeroes.Add(hero);
					if (!hero.invisibleBySkill)
					{
						if (hero.visible) hero.becomingInvis = true;
						else hero.becomingInvis = false;
						hero.visible = true;
					}
					else
					{
						if (hero.visible) hero.becomingInvis = true;
						else hero.becomingInvis = false;
						hero.visible = false;
					}
				}
			}

			List<Hero> hideout = new List<Hero>();
			List<int> teams = new List<int>();
			foreach (Bush bush in bushes)
			{
				hideout.Clear();
				teams.Clear();
				if (visibleHeroes.Count == 0) break;
				foreach (Hero hero in visibleHeroes)
				{
					if (hero.Distance(bush) <= bush.radius)
					{
						hideout.Add(hero);
						if (!teams.Contains(hero.team)) teams.Add(hero.team);
					}
				}
				if (teams.Count == 1)
				{
					foreach (Hero hero in hideout)
					{
						if (hero.visibilityTimer > 0) continue;
						hero.visible = false;
						visibleHeroes.Remove(hero);
					}
				}
			}
		}


		public bool isGameOver(List<Player> players)
		{
			foreach (Player player in players)
			{
				if (player.tower.isDead) return true;
				if (player.heroesAlive() == 0) return true;
			}

			return false;
		}

		internal Unit getUnitOfId(int id)
		{
			foreach (Unit unit in allUnits)
			{
				if (unit.id == id)
				{
					return unit;
				}
			}

			return null;
		}

		Event findNextEvent()
		{
			double firstTime = Const.ROUNDTIME;
			Event nextEvent = null;
			foreach (Event _event in events)
			{
				if (_event.t < firstTime && _event.t >= 0.0)
				{
					nextEvent = _event;
					firstTime = _event.t;
				}
			}

			return nextEvent;
		}

		public void beforeTurn(int turn, List<Player> players)
		{
			t = 0.0;
			round = turn;

			if (turn % Const.SPAWNRATE == Const.HEROCOUNT)
			{
				spawnUnits(players);
			}

			if (turn % Const.NEUTRALSPAWNRATE == Const.NEUTRALSPAWNTIME + Const.HEROCOUNT) spawnForestCreatures();
		}

		void spawnUnits(List<Player> players)
		{
			foreach (Player player in players)
			{
				for (int i = 0; i < Const.MELEE_UNIT_COUNT; i++)
				{
					allUnits.Add(Factories.generateUnit(0, players.IndexOf(player), i, player));
				}
				for (int i = 0; i < Const.RANGED_UNIT_COUNT; i++)
				{
					allUnits.Add(Factories.generateUnit(1, players.IndexOf(player), round / Const.SPAWNRATE % 3, player));
				}
			}
		}

		internal void initialize(List<Player> players)
		{
			setupGame(players);
			items = Factories.createItems(setupGold(players));
		}

		int setupGold(List<Player> players)
		{
			int amount = (int)(Const.random.NextDouble() * 551 + 100);
			foreach (Player p in players)
			{
				p.setGold(amount);
			}
			return amount;
		}

		void setupGame(List<Player> players)
		{
			allUnits.Add(Factories.generateTower(players[0], 0));
			allUnits.Add(Factories.generateTower(players[1], 1));

			if (Const.REMOVEFORESTCREATURES)
			{
				forestCampCount = 0;
			}
			else
			{
				spawns = MapFactory.generateSpawns();
				forestCampCount = spawns.Length;
				creatures = new Creature[forestCampCount];
				amplitude = new double[forestCampCount];
				for (int i = 0; i < forestCampCount; i++)
					amplitude[i] = 1.0;
			}

			List<Bush> tempBushes = MapFactory.generateBushes(spawns);

			// In lower leagues we ignore bushes. But lets add the visuals because it's pretty =)
			if (!Const.IGNOREBUSHES) this.bushes.AddRange(tempBushes);
		}


		// random spawn point placement
		internal CreatureSpawn[] spawns = { };
		Creature[] creatures = new Creature[0];
		double[] amplitude = { };

		void spawnForestCreatures()
		{

			for (int i = 0; i < forestCampCount; i++)
			{
				if (creatures[i] == null || creatures[i].isDead)
				{
					allUnits.Add(creatures[i] = Factories.generateCreature(spawns[i], amplitude[i]));
					amplitude[i] *= 1.2;
				}
			}
		}
	}
}