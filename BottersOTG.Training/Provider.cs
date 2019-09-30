using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BottersOTG.Training.Logging;
using Telogis.RouteCloud.GPUManagement;
using Telogis.RoutePlan.Logging;

namespace BottersOTG.Training {
	class Provider {
		public static readonly Logger Logger =
			new Logger(new DebugInterceptTextWriter(Console.Out), "service", "slot", "instance", formatter: CustomFormatters.Debug);

		public static readonly CudaManagerPool CudaManagerPool = new CudaManagerPool(Logger);
	}
}
