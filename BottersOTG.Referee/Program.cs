using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BOTG_Refree
{
    class CodePair
    {
        internal Code A;
        internal Code B;
        internal int A_p = 0;
        internal int B_p = 0;

        internal Code GetCode(int idx)
        {
            switch (idx)
            {
            case 0: return A;
            case 1: return B;
            default: throw new Exception();
            }
        }
        internal void AddPoints(int idx, int points)
        {
            switch (idx)
            {
            case 0: A_p += points; break;
            case 1: B_p += points; break;
            default: throw new Exception();
            }
        }
    }

    internal class Code
    {
        internal string file;
        internal string name;
        internal int score;
        internal int games;
        internal Code(string _file)
        {
            file = _file;
            name = Path.GetFileNameWithoutExtension(_file);
            score = 0;
            games = 0;
        }
    }

    class Program
    {
        static List<CodePair> Tumbles(List<Code> code)
        {
            List<CodePair> cps = new List<CodePair>();
            for (int a = 0; a < code.Count; a++)
            {
                for (int b = a + 1; b < code.Count; b++)
                {
                    CodePair cp = new CodePair();
                    cp.A = code[a];
                    cp.B = code[b];
                    cps.Add(cp);
                }
            }
            return cps;
        }

        static Player CreatePlayerState(int id, string botcode)
        {
            var process = new Process();
            if (Path.GetExtension(botcode).ToLower() == ".py")
            {
                process.StartInfo.FileName = "python.exe";
                process.StartInfo.Arguments = $"\"{botcode}\"";
            } else if (Path.GetExtension(botcode).ToLower() == ".exe")
            {
                process.StartInfo.FileName = $"\"{botcode}\"";
                process.StartInfo.Arguments = "";
            } else
            {
                throw new InvalidInputException($"Cannot handle \"{botcode}\" as a bot", botcode);
            }
            process.StartInfo.UseShellExecute = false;
            process.StartInfo.RedirectStandardOutput = true;
            process.StartInfo.RedirectStandardError = true;
            process.StartInfo.RedirectStandardInput = true;
            process.ErrorDataReceived += new DataReceivedEventHandler(OutputHandlerError);

            Player ps = new Player();
            ps.player_id = id;
            ps.pro = process;
            ps.code_file_name = botcode;

            process.Start();
            process.BeginErrorReadLine();

            return ps;
        }

        static void OutputHandlerError(object sendingProcess, DataReceivedEventArgs outLine)
        {
            //Console.WriteLine($"E: {outLine.Data}");
        }


        static int[][] permutation2 = { new[] { 0, 1 }, new[] { 1, 0 } };

        static string CodeDir = @"C:\temp\BotG\";
        static string DataDir = @"C:\temp\BotG\";
        static string BigRecordFileName = @"big_battle_log.txt";

        static void RunCodePair(CodePair cp, int loops)
        {
            string pair_log = Path.Combine(DataDir, $"battles_{cp.A.name}_{cp.B.name}.log");
            using (StreamWriter log = new StreamWriter(pair_log))
            {
                for (int i = 0; i < loops; i++)
                {
                    string seedtxt = "";
                    int seed;

                    foreach (var p in permutation2)
                    {
                        var res = Battle(i, cp.GetCode(p[0]).file, cp.GetCode(p[1]).file, log, seedtxt, out seed);
                        cp.AddPoints(p[0], res[0]);
                        cp.AddPoints(p[1], res[1]);

                        seedtxt = seed.ToString();
                        Console.WriteLine($"loop: {i}  a: {cp.A_p} b: {cp.B_p} {Path.GetFileNameWithoutExtension(cp.A.file)} {Path.GetFileNameWithoutExtension(cp.B.file)}");
                    }
                }

                string ll = $"Final: {cp.A_p} {cp.B_p} {cp.A.name} {cp.B.name}";
                log.WriteLine(ll);
                using (StreamWriter big_log = new StreamWriter(Path.Combine(DataDir, BigRecordFileName), true))
                {
                    big_log.WriteLine(ll);
                }
            }
        }

        static void Main(string[] args)
        {
            string[] code_files = {
                "simeon_1.py",
                "simeon_2.py",
            };

            List<Code> codes = new List<Code>();
            foreach (var cf in code_files)
            {
                codes.Add(new Code(Path.Combine(CodeDir, cf)));
            }

            foreach (var cp in Tumbles(codes))
            {
                RunCodePair(cp, 30);
            }
        }

        static int[] Battle(int loop, string codeA, string codeB, StreamWriter log, string use_seed, out int used_seed)
        {
            GameManager<Player> _gm = new GameManager<Player>();
            _gm.players.Add(CreatePlayerState(0, codeA));
            _gm.players.Add(CreatePlayerState(1, codeB));

            Referee _ref = new Referee();
            _ref.gameManager = _gm;

            Properties props = new Properties();
            if (use_seed != "")
                props.Add("seed", use_seed);

            _ref.init(props);

            int round = 0;
            while (_gm.game_ended == false && round < _gm.maxTurns)
            {
                _ref.gameTurn(round);

                round += 1;
            }
            used_seed = int.Parse(props.getProperty("seed", "0"));
            int sa = _gm.players[0].getScore();
            int sb = _gm.players[1].getScore();


            string lline = $"loop: {loop} seed: {used_seed} round: {round} a: {sa} b: {sb} {Path.GetFileNameWithoutExtension(codeA)} {Path.GetFileNameWithoutExtension(codeB)}";
            log.WriteLine(lline);

            foreach (var p in _gm.players)
            {
                p.pro.Kill();
                p.pro.WaitForExit();
                p.pro.Close();
            }

            int ra = 0, rb = 0;

            ra += sa >= sb ? 1 : 0;
            rb += sb >= sa ? 1 : 0;

            bat_res[0] = ra;
            bat_res[1] = rb;


            return bat_res;
        }

        static int[] bat_res = new int[] { 0, 0 };

    }


    public class Properties
    {
        Dictionary<string, string> values = new Dictionary<string, string>();
        public string getProperty(string name, string defaultValue)
        {
            string val;
            if (values.TryGetValue(name, out val))
                return val;
            return defaultValue;
        }

        public void Add(string name, string value)
        {
            values[name] = value;
        }
    }


    public class InvalidInputException : Exception
    {
        public InvalidInputException(string expected, object a) : base($"{expected} {a.ToString()}")
        {

        }
    }


	public class GameManager<T> where T : AbstractPlayer
	{
		public GameManager()
		{
			Const.InitTheThingsInConstThatAreNotConstant();
		}

        internal List<T> players = new List<T>();
        int league = 7;
        internal int maxTurns = 0;
        internal bool game_ended = false;

        // API's used by BotG
        public int getLeagueLevel() { return league; }
        public void setFrameDuration(int value) { }
        public void setMaxTurns(int value) { maxTurns = value; }
        public List<T> getActivePlayers() { return players.Where(p=>p.active).ToList();  }
        public List<T> getPlayers() { return players; }
        public void endGame() { game_ended = true; }
        public void addToGameSummary(string value) { System.Diagnostics.Debug.WriteLine($"SUM: {value}"); }

        // API's used to run game
    }

    public class AbstractPlayer
    {
        int score = 0;
		internal bool active = true;
        // BotG API's
        public string getNicknameToken() { return "Player"; }
        public void setScore(int value) { score = value; }
        public void deactivate(string message) { active = false; }
		public void sendInputLine(string message)
		{
			// Send "input" to the player pipe
			pro.StandardInput.WriteLine(message);
			if (player_id == 0)
				Debug.WriteLine(message);

        }
        public void execute() { }
        public string[] getOutputs(int lines_count)
        {
            // Read "output from player pipe
            var input = new List<string>();
            while (input.Count < lines_count)
            {
                string line = pro.StandardOutput.ReadLine();
                if (line != null && line.Length > 0)
                    input.Add(line);
            }
            return input.ToArray();
        }
        public int getIndex() { return player_id; }
        public int getScore() { return score; }

        // My API's
        internal Process pro;
        internal int player_id;
        internal string code_file_name;
    }
}
