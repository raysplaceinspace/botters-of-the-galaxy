using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BottersOTG.Intelligence.Decisions;
using BottersOTG.Model;

namespace BottersOTG.Training {
	static class PolicyHelper {
		public static uint CategoriesToBits(IEnumerable<Enum> categories) {
			uint bits = 0;
			foreach (Enum category in categories) {
				bits |= (uint)1 << Convert.ToInt32(category);
			}
			return bits;
		}

		public static List<Enum> BitsToCategories(CategoricalAxis axis, uint categoryBits) {
			List<Enum> categories = new List<Enum>();
			int index = 0;
			while (categoryBits > 0) {
				if ((categoryBits & 1) != 0) {
					categories.Add(CategoricalAxisValue(axis, index));
				}
				categoryBits >>= 1;
				++index;
			}
			return categories;
		}

		private static Enum CategoricalAxisValue(CategoricalAxis axis, int index) {
			switch (axis) {
				case CategoricalAxis.MyHero:
				case CategoricalAxis.MyHeroes:
				case CategoricalAxis.EnemyHeroes:
					return (HeroType)index;
				case CategoricalAxis.MySpells:
				case CategoricalAxis.AllySpells:
				case CategoricalAxis.EnemySpells:
					return (ActionType)index;
				case CategoricalAxis.IsVisible:
					return (CategoricalBoolean)index;
				default: throw new ArgumentException("Cannot convert axis value from integer: " + axis);
			}
		}

	}
}
