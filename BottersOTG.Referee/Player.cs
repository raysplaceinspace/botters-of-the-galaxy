using System;
using System.Collections.Generic;
using System.Linq;

namespace BOTG_Refree
{
	public class Player : AbstractPlayer {
		public List<Hero> heroes = new List<Hero>();
		public Tower tower;
		public int gold;
		public int denies;
		public int unitKills;

		public void addHero(Hero hero) {
			this.heroes.Add(hero);
		}

		public void handlePlayerOutputs(string[] outputs) {
			List<Hero> internalHeroes = new List<Hero>(heroes);
			if (outputs.Length == 2 && outputs[1].StartsWith("SELL") && heroesAlive() == 2) { 
                //Flip order, so SELL happens first, thus the gold can be used by other player
				Hero first = internalHeroes[0];
				internalHeroes.RemoveAt(0);
				internalHeroes.Add(first);

				string temp = outputs[0];
				outputs[0] = outputs[1];
				outputs[1] = temp;
			}

			int outputCounter = 0;
			for (int i = 0; i < internalHeroes.Count; i++) {
				Hero hero = internalHeroes[i];
				if (hero.isDead) {
					continue;
				}

				if (hero.stunTime > 0) {
					outputCounter++;
					continue;
				}

				string roundOutput = outputs[outputCounter++];
				doHeroCommand(roundOutput, hero);
			}
		}

		static char[] delimiter = ";".ToCharArray();



        private void doHeroCommand(string roundOutput, Hero hero) {
			string[] roundOutputSplitted = roundOutput.Split(delimiter, 2);

			try {

				string message = roundOutputSplitted.Length > 1 ? roundOutputSplitted[1] : "";
				string[] outputValues = roundOutputSplitted[0].Split(' ');
				string command = outputValues[0];
				int arguments = outputValues.Length - 1;

				// Verification
				bool allNumbers = true;
				for (int num = 1; num < outputValues.Length; num++) {
					if (!Utilities.isNumber(outputValues[num])) allNumbers = false;
				}

				if (command == "MOVE_ATTACK" && arguments == 3 && allNumbers) {
					// MOVE_ATTACK x y unitID
					double x = double.Parse(outputValues[1]);
					double y = double.Parse(outputValues[2]);
					int id = int.Parse(outputValues[3]);
					Point target = new Point(x, y);
					hero.runTowards(target);
					Unit unit = Const.game.getUnitOfId(id);
					if (unit != null) {
						Const.game.events.Add(new DelayedAttackEvent(unit, hero, Utilities.timeToReachTarget(hero, target, hero.moveSpeed)));
					} else printError("Can't attack unit of id: " + id);
				}

				else if (command == "MOVE" && arguments == 2 && allNumbers) {
					double x = double.Parse(outputValues[1]);
					double y = double.Parse(outputValues[2]);
					hero.runTowards(new Point(x, y));
				}

				else if (command == "ATTACK_NEAREST" && arguments == 1 && !allNumbers) {
					string unitType = outputValues[1];
					Unit toHit = hero.findClosestOnOtherTeam(unitType);
					if (toHit != null) {
						hero.attackUnitOrMoveTowards(toHit, 0.0);
					}
				}

				else if (command == "ATTACK" && arguments == 1 && allNumbers) {
					int id = int.Parse(outputValues[1]);
					Unit unit = Const.game.getUnitOfId(id);
					if (unit != null && hero.allowedToAttack(unit)) {
						hero.attackUnitOrMoveTowards(unit, 0.0);
					} else printError("Cant attack: " + id);

				}

				else if (command == "BUY" && arguments == 1 && !allNumbers) {
					Item item = Const.game.items[outputValues[1]];

					if (item == null) {
						printError(" tried to buy item: " + outputValues[1] + ", but it does not exist");
					} else if (gold < item.cost) {
						printError(" can't afford " + outputValues[1]);
					} else if (hero.items.Count >= Const.MAXITEMCOUNT) {
						printError("Can't have more than " + Const.MAXITEMCOUNT + " items. " + (item.isPotion ? "Potions need a free item slot to be bought." : ""));
					} else {
						hero.addItem(item);
						gold -= item.cost;
					}
				}

				else if (command == "SELL" && arguments == 1 && !allNumbers) {
					string itemName = outputValues[1];
					var foundItem = hero.items.Where(currItem=>currItem.name==itemName).First();

					if (foundItem == null) {
						printError("Selling not owned item " + foundItem.name);
					} else {
						hero.removeItem(foundItem);
						gold += Utilities.round(foundItem.cost * Const.SELLITEMREFUND);
					}
				}

				else if (command == "WAIT" && arguments == 0) {
					return;
				}

				else if (allNumbers)
				{ // skillz
					foreach (SkillBase skill in hero.skills) {
						if (skill == null) continue;
						if (command == skill.skillName) {
							if (skill.manaCost > hero.mana) {
								printError("Not enough mana to use " + skill.skillName);
							} else if (skill.cooldown > 0) {
								printError("Skill on cooldown: " + skill.skillName + ". Cooldown left: " + skill.cooldown);
							} else {
								double x = -1;
								double y = -1;
								int unitId = -1;
								if (outputValues.Length == 3 && skill.getTargetType() == SkillType.POSITION) {
									x = double.Parse(outputValues[1]);
									y = double.Parse(outputValues[2]);
								}
								else if (outputValues.Length == 2 && skill.getTargetType() == SkillType.UNIT) {
									unitId = int.Parse(outputValues[1]);
									Unit unit = Const.game.getUnitOfId(unitId);

									if (!hero.allowedToTarget(unit) || unit is Tower) {
										printError(hero.heroType + " can't target unit with spell. Either invisible or not existing.");
										return;
									}
								}
								else if (outputValues.Length == 1 && skill.getTargetType() == SkillType.SELF) { }
								else {
									printError(hero.heroType + " invalid number of parameters on spell. " + roundOutputSplitted[0]);
									return;
								}

								hero.mana -= skill.manaCost;
								skill.cooldown = skill.initialCooldown;
								hero.invisibleBySkill = false;
								skill.doSkill(Const.game, x, y, unitId);
							}

							return;
						}
					}

					printError(" tried to use a spell not found on  " + hero.heroType + ". Input was: " + roundOutputSplitted[0]);
				} else {
					printError(" tried to use an invalid command. Invalid parameters or name. Command was: " + roundOutputSplitted[0]);
				}
			} catch (Exception /*e*/)
			{
				printError(" tried to use an invalid command. Invalid parameters or name. Command was: " + roundOutputSplitted[0]);
			}
		}

		public int heroesAlive() {
			int alive = 0;
			foreach (Hero hero in heroes) {
				if (!hero.isDead) alive++;
			}
			return alive;
		}

		public int getExpectedOutputLines() {
			if (this.heroes.Count < Const.HEROCOUNT) return 1;
			return this.heroes.Where(h=> !h.isDead).Count();
		}

		public int getGold() {
			return this.gold;
		}

		public void setGold(int amount) {
			this.gold = amount;
		}

		private void printError(string message) {
			Console.WriteLine(message);
		}
	}
}
