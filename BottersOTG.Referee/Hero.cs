using System;
using System.Collections.Generic;
using System.Reflection;

namespace BOTG_Refree
{

    public class Hero : Unit {

        public SkillBase[] skills = new SkillBase[Const.SKILLCOUNT];
        public string heroType;
        public int mana;
        public int manaregeneration;
        public int maxMana;
        public int visibilityTimer;
        public List<Item> items = new List<Item>();

        public Hero(double x, double y, int health, int team, int moveSpeed, Player player, string heroType) : base(x, y, health, team, moveSpeed, player)
        {
            this.goldValue = Const.HERO_GOLD_VALUE;
            this.heroType = heroType;
        }

        override internal void afterRound() {
            base.afterRound();
            visibilityTimer = Math.Max(0, visibilityTimer - 1);
            mana += manaregeneration;
            if (mana > maxMana) mana = maxMana;
            foreach(SkillBase skill in skills) {
                if (skill == null) continue;
                skill.cooldown = Math.Max(0, skill.cooldown - 1);
            }
        }

        internal void addItem(Item item) {
            if (item == null) {
                return;
            }

            addCharacteristics(item, 1);
            if (!item.isPotion) {
                items.Add(item);
            }
        }

		internal override void findAction(List<Unit> allUnits)
		{
			// this function is for Creatures, Towers, LaneUnits, not Hero's
		}

		internal void removeItem(Item item) {
            if (!items.Contains(item)) return;
            items.Remove(item);
            addCharacteristics(item, -1);
        }

        void addCharacteristics(Item item, int amplitude) {
            var characteristics = item.stats;
            var c = this.GetType();
            foreach( var kv in characteristics) {
                try {
                    FieldInfo f = c.GetField(kv.Key);
                    f.SetValue(this, (int)f.GetValue(this) + kv.Value * amplitude);
                } catch (Exception /*e*/) {
                }
            }

            var c2 = this.GetType().BaseType;
            foreach (var kv in characteristics)
            {
                try {
                    FieldInfo f = c2.GetField(kv.Key);
					if (f != null)
					{
						f.SetValue(this, (int)f.GetValue(this) + kv.Value * amplitude);
					}
                } catch (Exception /*e*/) {
                }
            }

            if (mana > maxMana) mana = maxMana;
            if (health > maxHealth) health = maxHealth;
            if (mana < 0) mana = 0;
            if (health <= 0) health = 1;
            if (moveSpeed > Const.MAXMOVESPEED) moveSpeed = Const.MAXMOVESPEED;
        }

        public bool isVisible(Player player) {
            return (!isDead && (visible || player == this.player));
        }

        override
        public string getType() {
            return "HERO";
        }

        override
        protected string getExtraProperties() {
            return
                skills[0].cooldown + " " +
                skills[1].cooldown + " " +
                skills[2].cooldown + " " +
                mana + " " +
                maxMana + " " +
                manaregeneration + " " +
                heroType + " " +
                (visible ? 1 : 0) + " " +
                items.Count;
        }
    }

}