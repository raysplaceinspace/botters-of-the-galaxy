using System;

namespace BOTG_Refree
{
	public class Point
	{
		public double x;
		public double y;

		public Point(double x, double y)
		{
			this.x = x;
			this.y = y;
		}

		public double Distance(Point p)
		{
			return Math.Sqrt(Distance2(p));
		}

		public double Distance2(Point p)
		{
			return ((this.x - p.x) * (this.x - p.x) + (this.y - p.y) * (this.y - p.y));
		}
	}
}
