using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BottersOTG.Model;

namespace BottersOTG.Training {
	public class Matchup {
		public HeroType[] Team0;
		public HeroType[] Team1;

		public override string ToString() {
			return
				string.Join("/", Team0.Select(x => x.ToString())) +
				" vs " +
				string.Join("/", Team1.Select(x => x.ToString()));
		}
	}
}
