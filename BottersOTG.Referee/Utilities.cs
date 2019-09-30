using System;
using System.Collections.Generic;

namespace BOTG_Refree
{
	public class Utilities
	{
		public static bool isNumber(String s)
		{
			double d;
			bool isnum = double.TryParse(s, out d);
			return isnum && double.IsInfinity(d) == false && double.IsNaN(d) == false;
		}

		public static int rndInt(int min, int max)
		{
			return (int)(Const.random.NextDouble() * (max - min) + min);
		}

		public static int round(double x)
		{
			int s = x < 0 ? -1 : 1;
			return s * (int)((s * x) + 0.5);
		}


		public static double timeToReachTarget(Point start, Point stop, double speed)
		{
			return start.Distance(stop) / speed;
		}
	}
}