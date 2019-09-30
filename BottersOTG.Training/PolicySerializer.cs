using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BottersOTG.Intelligence.Decisions;
using BottersOTG.Model;

namespace BottersOTG.Training {
	public static class PolicySerializer {
		public static string Serialize(Policy policy, char indentChar = '\t') {
			CodeWriter writer = new CodeWriter() {
				IndentChar = indentChar,
			};

			writer.AppendLine("new Policy {{");
			using (writer.Indent()) {
				Serialize("HeroMatchups = ", policy.HeroMatchups, ",", writer);
				Serialize("Default = ", policy.Default, ",", writer);
				Serialize("Deadpool = ", policy.Deadpool, ",", writer);
				Serialize("DoctorStrange = ", policy.DoctorStrange, ",", writer);
				Serialize("Hulk = ", policy.Hulk, ",", writer);
				Serialize("Ironman = ", policy.Ironman, ",", writer);
				Serialize("Valkyrie = ", policy.Valkyrie, ",", writer);
			}
			writer.AppendLine("}};");

			return writer.ToString();
		}

		private static void Serialize(string prefix, HeroChoices choices, string suffix, CodeWriter writer) {
			writer.AppendLine("{0}new HeroChoices {{", prefix);
			using (writer.Indent()) {
				writer.AppendLine("PrimaryChoice = {0},", SerializeEnum(choices.PrimaryChoice));
				writer.AppendLine("SecondaryChoices = new Dictionary<HeroType, HeroType> {{", choices.PrimaryChoice.ToString());
				using (writer.Indent()) {
					foreach (KeyValuePair<HeroType, HeroType> choice in choices.SecondaryChoices) {
						writer.AppendLine("{{ {0}, {1} }},", SerializeEnum(choice.Key), SerializeEnum(choice.Value));
					}
				}
				writer.AppendLine("}},");
			}
			writer.AppendLine("}}{0}", suffix);
		}

		private static void Serialize(string prefix, IDecisionNode node, string suffix, CodeWriter writer) {
			if (node == null) {
				writer.AppendLine("{0}null{1}", prefix, suffix);
			} else if (node is DecisionLeaf) {
				Serialize(prefix, (DecisionLeaf)node, suffix, writer);
			} else if (node is DecisionNode) {
				Serialize(prefix, (DecisionNode)node, suffix, writer);
			} else {
				writer.AppendLine("{0}<{1}>{2}", prefix, node, suffix);
			}
		}

		private static void Serialize(string prefix, DecisionLeaf leaf, string suffix, CodeWriter writer) {
			List<string> arguments = new List<string> { SerializeString(leaf.DefaultTactic.ToString()) };
			if (leaf.PerHeroTactics != null) {
				foreach (KeyValuePair<HeroType, Tactic> kvp in leaf.PerHeroTactics) {
					arguments.Add(HeroTypeToCode(kvp.Key) + ":" + SerializeString(kvp.Value.ToString()));
				}
			}
			writer.AppendLine("{0}L({1}){2}", prefix, string.Join(", ", arguments), suffix);
		}

		private static string HeroTypeToCode(HeroType heroType) {
			switch (heroType) {
				case HeroType.Deadpool: return "d";
				case HeroType.DoctorStrange: return "s";
				case HeroType.Hulk: return "h";
				case HeroType.Ironman: return "i";
				case HeroType.Valkyrie: return "v";
				default: throw new ArgumentException("Unknown hero type: " + heroType);
			}
		}

		private static void Serialize(string prefix, DecisionNode splitter, string suffix, CodeWriter writer) {
			writer.AppendLine("{0}N({1},", prefix, SerializePartitioner(splitter.Partitioner));
			using (writer.Indent()) {
				Serialize("", splitter.Left, ",", writer);
				Serialize("", splitter.Right, ")" + suffix, writer);
			}
		}

		private static string SerializePartitioner(IPartitioner partitioner) {
			if (partitioner is ContinuousPartitioner) {
				return SerializePartitioner((ContinuousPartitioner)partitioner);
			} else if (partitioner is CategoricalPartitioner) {
				return SerializePartitioner((CategoricalPartitioner)partitioner);
			} else {
				return string.Format("<{0}>", partitioner);
			}
		}

		private static string SerializePartitioner(ContinuousPartitioner partitioner) {
			return string.Format("D({0}, {1:0.0##})", SerializeString(partitioner.Axis.ToString()), partitioner.Split);
		}

		private static string SerializePartitioner(CategoricalPartitioner partitioner) {
			return string.Format(
				"C({0}, {1})",
				SerializeString(partitioner.Axis.ToString()),
				string.Join(", ", partitioner.Categories.Select(SerializeEnum)));
		}

		private static string SerializeEnum(Enum v) {
			return v.GetType().Name + "." + v.ToString();
		}

		private static string SerializeString(string str) {
			return '"' + str + '"';
		}
	}
}
