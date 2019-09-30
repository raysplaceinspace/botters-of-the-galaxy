using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace BottersOTG.Training {
	public static class EmbeddedResourceHelper {
		private const string EmbeddedResourcePrefix = "BottersOTG.Training";

		public static Stream OpenResourceStream(string resourceName) {
			Assembly assembly = Assembly.GetExecutingAssembly();

			Stream result = assembly.GetManifestResourceStream(EmbeddedResourcePrefix + "." + resourceName);
			if (result == null) {
				throw new ArgumentException(string.Format(
					"No embedded resource exists with the name '{0}' - have you set its Build Action to Embedded Resource?",
					resourceName));
			}
			return result;
		}

		public static string GetResource(string resourceName) {
			using (Stream stream = OpenResourceStream(resourceName)) {
				using (StreamReader reader = new StreamReader(stream)) {
					return reader.ReadToEnd();
				}
			}
		}

		public static byte[] GetResourceBytes(string resourceName) {
			using (Stream stream = OpenResourceStream(resourceName)) {
				if (stream != null) {
					byte[] result = new byte[stream.Length];
					stream.Read(result, 0, result.Length);
					return result;
				} else {
					return null;
				}
			}
		}
	}
}
