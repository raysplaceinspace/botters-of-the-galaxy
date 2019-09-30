namespace BOTG_Refree
{
    public class Damage
    {
        public Damage(Unit target, Unit attacker, int damage)
        {
            this.target = target;
            this.attacker = attacker;
            this.damage = damage;
        }
        public Unit target;
        public Unit attacker;
        public int damage;
    }
}
