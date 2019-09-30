using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BottersOTG.Model;

namespace BottersOTG.Training {
	class Rollout {
		public Matchup Matchup;
		public double WinRate;
		public List<RolloutTick> Ticks;
		public World FinalWorld;
		public IntermediateEvaluator.Evaluation FinalEvaluation;
	}
}
