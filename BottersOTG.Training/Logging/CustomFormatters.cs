using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BottersOTG.Training.Logging {
	public static class CustomFormatters {
		public static string Debug(IDictionary<string, object> obj) {
			DateTime timestamp = ((DateTime)obj["timestamp"]);
			string level = obj["level"].ToString().ToUpperInvariant();
			string message = obj["message"].ToString();

			if (obj.ContainsKey("exception") && obj["exception"] is Exception && new[] { "WARN", "ERROR", "FATAL" }.Contains(level)) {
				message = string.Join("\n", FormatException((Exception)obj["exception"]));
			}

			return string.Format("[{0}] {1}: {2}", timestamp.ToLocalTime().ToString("HH:mm:ss.fff"), level, message);
		}

		public static string GetExceptionFormatted(Exception ex) {
			return string.Join("\n", FormatException(ex));
		}

		private static IEnumerable<string> FormatException(Exception ex) {
			yield return string.Format("({0}) {1}", ex.GetType().FullName, ex.Message);

			if (ex is AggregateException) {
				yield return " | Aggregate Exceptions:";
				foreach (var line in ((AggregateException)ex).InnerExceptions.SelectMany(FormatException)) {
					yield return " | " + line;
				}
			} else if (ex.InnerException != null) {
				yield return " | Inner Exceptions:";
				foreach (var line in FormatException(ex.InnerException)) {
					yield return " | " + line;
				}
			}

			if (!string.IsNullOrEmpty(ex.StackTrace)) {
				yield return " | StackTrace:";
				foreach (var line in ex.StackTrace.Split('\n')) {
					yield return " |" + line;
				}
			}

			if (ex.Data != null && ex.Data.Count > 0) {
				yield return " | Data:";
				foreach (DictionaryEntry kvp in ex.Data) {
					yield return $" |   {kvp.Key}: {kvp.Value}";
				}
			}
		}
	}
}
