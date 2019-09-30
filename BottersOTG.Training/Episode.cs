using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BottersOTG.Intelligence.Decisions;
using BottersOTG.Model;

namespace BottersOTG.Training {
	public class Episode {
		public World World;
		public Unit Hero;
		public Tactic Tactic;
		public double Weight;
	}
}
