using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BottersOTG.Training.Logging {
	class DebugInterceptTextWriter : TextWriter {
		private readonly TextWriter _passthrough;
		public override Encoding Encoding => Encoding.UTF8;

		public DebugInterceptTextWriter(TextWriter passthrough) {
			_passthrough = passthrough;
		}

		public override void Write(char value) {
			_passthrough.Write(value);
			Debug.Write(value.ToString());
		}

		public override void WriteLine(string value) {
			_passthrough.WriteLine(value);
			Debug.WriteLine(value);
		}
	}
}
