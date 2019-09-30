namespace BOTG_Refree
{
    public class Const
    {
        public const int MAXINT = int.MaxValue;

        public static System.Random random;
		//Container
		public static Game game;

        public static bool REMOVEFORESTCREATURES;
        public static bool IGNOREITEMS;
        public static bool IGNORESKILLS;
        public static double TOWERHEALTHSCALE;
        public static bool IGNOREBUSHES;
        public static int TOWERDAMAGE;
		public static int Rounds; 
		public static int HEROCOUNT; 
		public static int MELEE_UNIT_COUNT; 
		public static int RANGED_UNIT_COUNT;

		public static int GLOBAL_ID;

		public static void InitTheThingsInConstThatAreNotConstant()
		{
			game = new Game();

			REMOVEFORESTCREATURES = false;
			IGNOREITEMS = false;
			IGNORESKILLS = false;
			TOWERHEALTHSCALE = 1.0;
			IGNOREBUSHES = false;
			TOWERDAMAGE = 190;
			Rounds = 250;
			HEROCOUNT = 2;
			MELEE_UNIT_COUNT = 3;
			RANGED_UNIT_COUNT = 1;

			GLOBAL_ID = 1;
		}

		//MISC
		public const double EPSILON = 0.00001;
        public const double ROUNDTIME = 1.0;
        public const int MAPWIDTH = 1920;
        public const int MAPHEIGHT = 780;
        public const int MAXITEMCOUNT = 4;
        public const double SELLITEMREFUND = 0.5; 

        //TEAM0
        public readonly static Point TOWERTEAM0 = new Point(100, 540);
        public readonly static Point SPAWNTEAM0 = new Point(TOWERTEAM0.x + 60, TOWERTEAM0.y - 50);
        public readonly static Point HEROSPAWNTEAM0 = new Point(TOWERTEAM0.x + 100, TOWERTEAM0.y + 50);
        public readonly static Point HEROSPAWNTEAM0HERO2 = new Point(HEROSPAWNTEAM0.x, TOWERTEAM0.y - 50);

        //TEAM1
        public readonly static Point TOWERTEAM1 = new Point(MAPWIDTH - TOWERTEAM0.x, TOWERTEAM0.y);
        public readonly static Point SPAWNTEAM1 = new Point(MAPWIDTH - SPAWNTEAM0.x, SPAWNTEAM0.y);
        public readonly static Point HEROSPAWNTEAM1 = new Point(MAPWIDTH - HEROSPAWNTEAM0.x, HEROSPAWNTEAM0.y);
        public readonly static Point HEROSPAWNTEAM1HERO2 = new Point(HEROSPAWNTEAM1.x, HEROSPAWNTEAM0HERO2.y);

        //HERO
        public const int SKILLCOUNT = 3;
        public const int MAXMOVESPEED = 450;

        //UNIT
        public const int SPAWNRATE = 15;
        public const int UNITTARGETDISTANCE = 400;
        public const int UNITTARGETDISTANCE2 = UNITTARGETDISTANCE * UNITTARGETDISTANCE;
        public const int AGGROUNITRANGE = 300;
        public const int AGGROUNITRANGE2 = AGGROUNITRANGE * AGGROUNITRANGE;
        public const int AGGROUNITTIME = 3;
        public const double DENYHEALTH = 0.4;

        public const double BUSHRADIUS = 50;

        //TOWERS
        public const int TOWERHEALTH = 3000;

        //NEUTRAL CREEP
        public const int NEUTRALSPAWNTIME = 4;
        public const int NEUTRALSPAWNRATE = 40;
        public const int NEUTRALGOLD = 100;

        //SPELLS
        // KNIGHT
        public const double EXPLOSIVESHIELDRANGE2 = 151 * 151;
        public const int EXPLOSIVESHIELDDAMAGE = 50;

        // LANCER
        public const double POWERUPDAMAGEINCREASE = 0.3;

        //GOLD UNIT VALUES
        public const int MELEE_UNIT_GOLD_VALUE = 30;
        public const int RANGER_UNIT_GOLD_VALUE = 50;
        public const int HERO_GOLD_VALUE = 300;

        //ITEM STATS
        public const string DAMAGE = "damage";
        public const string HEALTH = "health";
        public const string MAXHEALTH = "maxHealth";
        public const string MAXMANA = "maxMana";
        public const string MANA = "mana";
        public const string MOVESPEED = "moveSpeed";
        public const string MANAREGEN = "manaregeneration";
        public static string[] STATS = { DAMAGE, HEALTH, MANA, MOVESPEED, MANAREGEN };
        public const int MINIMUM_STAT_COST = 30;
        public const int NB_ITEMS_PER_LEVEL = 5;

    }
}
