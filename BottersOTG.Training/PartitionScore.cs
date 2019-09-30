using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BottersOTG.Intelligence.Decisions;

namespace BottersOTG.Training {
	public class PartitionScore {
		public IPartitioner Partitioner;

		public List<Episode> LeftEpisodes;
		public List<Episode> RightEpisodes;

		public double Entropy;
	}
}
