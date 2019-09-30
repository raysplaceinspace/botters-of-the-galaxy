using System.Collections.Generic;

namespace BOTG_Refree
{
    public class Item
    {
        public string name;
        public Dictionary<string, int> stats;
        public int cost;
        public bool isPotion;

        public Item(string name, Dictionary<string, int> stats, int cost, bool isPotion)
        {
            this.name = name;
            this.stats = fillEmptyStats(stats);
            this.cost = cost;
            this.isPotion = isPotion;
        }

        internal string getPlayerString()
        {
            return name +
                    " " + cost +
                    " " + stats[Const.DAMAGE] +
                    " " + stats[Const.HEALTH] +
                    " " + stats[Const.MAXHEALTH] +
                    " " + stats[Const.MANA] +
                    " " + stats[Const.MAXMANA] +
                    " " + stats[Const.MOVESPEED] +
                    " " + stats[Const.MANAREGEN] +
                    " " + (isPotion ? 1 : 0);
        }

        static Dictionary<string, int> fillEmptyStats(Dictionary<string, int> stats)
        {
            if (!stats.ContainsKey(Const.DAMAGE)) stats.Add(Const.DAMAGE, 0);
            if (!stats.ContainsKey(Const.HEALTH)) stats.Add(Const.HEALTH, 0);
            if (!stats.ContainsKey(Const.MAXHEALTH)) stats.Add(Const.MAXHEALTH, 0);
            if (!stats.ContainsKey(Const.MANA)) stats.Add(Const.MANA, 0);
            if (!stats.ContainsKey(Const.MAXMANA)) stats.Add(Const.MAXMANA, 0);
            if (!stats.ContainsKey(Const.MOVESPEED)) stats.Add(Const.MOVESPEED, 0);
            if (!stats.ContainsKey(Const.MANAREGEN)) stats.Add(Const.MANAREGEN, 0);
            return stats;
        }
    }
}
