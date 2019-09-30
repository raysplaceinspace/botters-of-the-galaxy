using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BottersOTG.Training {
	public class CodeWriter {
		private readonly StringBuilder _sb = new StringBuilder();
		private int _indent = 0;

		public char IndentChar = '\t';

		public void AppendLine(string format, params object[] objs) {
			_sb.Append(IndentChar, _indent);
			_sb.AppendFormat(format, objs);
			_sb.AppendLine();
		}

		public override string ToString() {
			return _sb.ToString();
		}

		public IDisposable Indent() {
			return new IndentedScope(this);
		}

		private class IndentedScope : IDisposable {
			public readonly CodeWriter Parent;

			public IndentedScope(CodeWriter parent) {
				Parent = parent;
				++Parent._indent;
			}

			public void Dispose() {
				--Parent._indent;
			}
		}
	}
}
