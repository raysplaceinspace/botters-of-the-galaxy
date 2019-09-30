
namespace BOTG_Refree
{
    public class Bush : Point
    {
        internal double radius;
        public Bush(double x, double y) : base(x, y)
        {
            this.radius = Const.BUSHRADIUS;
        }

        public string getPlayerString()
        {
            return "BUSH" + " " + (int)x + " " + (int)y + " " + (int)radius;
        }
    }
}