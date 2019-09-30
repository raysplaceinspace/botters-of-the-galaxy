using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BottersOTG.Intelligence.Decisions;
using BottersOTG.Model;

namespace BottersOTG.Training {
	public class RolloutTick {
		public World World;
		public Dictionary<int, Tactic> HeroTactics;
	}
}
