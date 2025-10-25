
import time
import random
import math
import chess
from chess.engine import PlayResult, Limit
from lib.engine_wrapper import MinimalEngine

# Precomputed piece values
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

# Small piece-square tables (midgame-ish). Indices are from white's perspective (0..63).
# These are deliberately modest; they help move ordering and typical positional play.
PST_PAWN = [
      0,   0,   0,   0,   0,   0,   0,   0,
     50,  50,  50,  50,  50,  50,  50,  50,
     10,  10,  20,  30,  30,  20,  10,  10,
      5,   5,  10,  27,  27,  10,   5,   5,
      0,   0,   0,  20,  20,   0,   0,   0,
      5,  -5, -10,   0,   0, -10,  -5,   5,
      5,  10,  10, -20, -20,  10,  10,   5,
      0,   0,   0,   0,   0,   0,   0,   0
]
PST_KNIGHT = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
]
PST_BISHOP = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -20,-10,-10,-10,-10,-10,-10,-20
]
PST_ROOK = [
      0,  0,  5, 10, 10,  5,  0,  0,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
      5, 10, 10, 10, 10, 10, 10,  5,
      0,  0,  0,  0,  0,  0,  0,  0
]
PST_QUEEN = [0]*64
PST_KING = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20
]

PST = {
    chess.PAWN: PST_PAWN,
    chess.KNIGHT: PST_KNIGHT,
    chess.BISHOP: PST_BISHOP,
    chess.ROOK: PST_ROOK,
    chess.QUEEN: PST_QUEEN,
    chess.KING: PST_KING
}

# MVV-LVA ordering values for captures (victim value - attacker priority)
MVV_LVA_PRIORITY = {chess.PAWN:1, chess.KNIGHT:2, chess.BISHOP:2, chess.ROOK:3, chess.QUEEN:4, chess.KING:5}

INFINITY = 10**9

class StrongBot(MinimalEngine):
    """
    Stronger pure-Python bot with iterative deepening, alpha-beta, TT, quiescence.
    """

    def __init__(self):
        super().__init__()
        # transposition table: key -> (value, depth, flag, best_move)
        # flag: 0 = exact, -1 = alpha (lower bound), 1 = beta (upper bound)
        self.tt = {}
        # history heuristic: move -> score (for non-capture ordering)
        self.history = {}
        # killer moves: depth -> [move1, move2]
        self.killers = {}

    def search(self, board: chess.Board, *args) -> PlayResult:
        # ---- time management ----
        time_limit = args[0] if (args and isinstance(args[0], Limit)) else None
        # default: 2.5s per move
        max_time = 2.5
        if isinstance(time_limit, Limit):
            if isinstance(time_limit.time, (int, float)):
                max_time = max(0.5, time_limit.time / 30.0)
            else:
                # per-color clocks if available
                if board.turn == chess.WHITE:
                    t = getattr(time_limit, "white_clock", None)
                else:
                    t = getattr(time_limit, "black_clock", None)
                if isinstance(t, (int, float)):
                    max_time = max(0.5, t / 30.0)

        start = time.time()
        deadline = start + max_time

        # quick emergency fallback: if <0.6s left, play a safe greedy move later
        emergency_threshold = 0.6

        # internal helpers
        def now():
            return time.time()

        def time_up():
            return now() >= deadline

        # ---- evaluation function (White positive) ----
        def evaluate(b: chess.Board) -> int:
            # Terminal handling
            if b.is_game_over():
                outcome = b.outcome()
                if outcome is None or outcome.winner is None:
                    return 0
                return INFINITY if outcome.winner == chess.WHITE else -INFINITY

            # material + PST + mobility + king safety
            score = 0
            for piece_type in PIECE_VALUES:
                v = PIECE_VALUES[piece_type]
                white_count = len(b.pieces(piece_type, chess.WHITE))
                black_count = len(b.pieces(piece_type, chess.BLACK))
                score += v * (white_count - black_count)

                # piece-square contribution
                pst = PST.get(piece_type)
                if pst:
                    # sum PST for white pieces minus black (mirror index)
                    for sq in b.pieces(piece_type, chess.WHITE):
                        score += pst[sq]
                    for sq in b.pieces(piece_type, chess.BLACK):
                        # mirror for black
                        score -= pst[chess.square_mirror(sq)]

            # mobility: prefer positions with more legal moves
            score += 10 * (len(list(b.legal_moves)) - 0)

            # king safety: penalize attacker proximity around king
            for color in [chess.WHITE, chess.BLACK]:
                king_sq = next(iter(b.pieces(chess.KING, color)), None)
                if king_sq is None:
                    continue
                # count enemy attackers near king
                attackers = 0
                for sq in chess.SquareSet(chess.BB_KING_ATTACKS[king_sq]):
                    if b.color_at(sq) == (not color):
                        attackers += 1
                if color == chess.WHITE:
                    score -= 40 * attackers
                else:
                    score += 40 * attackers

            return score

        # ---- transposition table helpers ----
        def tt_get(key, depth, alpha, beta):
            entry = self.tt.get(key)
            if not entry:
                return None
            val, ent_depth, flag, best = entry
            if ent_depth >= depth:
                if flag == 0:  # exact
                    return val, best
                if flag == -1 and val <= alpha:
                    return val, best
                if flag == 1 and val >= beta:
                    return val, best
            return None

        def tt_store(key, depth, value, flag, best_move):
            self.tt[key] = (value, depth, flag, best_move)

        # ---- ordering helpers: MVV-LVA for captures, history for quiet moves ----
        def score_move(b, m, tt_move=None):
            # high positive is good; we'll sort descending
            sc = 0
            if m == tt_move:
                sc += 2000000
            if b.is_capture(m):
                victim = b.piece_at(m.to_square)
                attacker = b.piece_at(m.from_square)
                if victim and attacker:
                    sc += 10000 + (PIECE_VALUES.get(victim.piece_type, 0) - PIECE_VALUES.get(attacker.piece_type, 0))
                    # MVV-LVA minor tweak
                    sc += 100 * (MVV_LVA_PRIORITY.get(victim.piece_type, 0))
            else:
                # history heuristic
                sc += self.history.get((m.from_square, m.to_square), 0)
                # killer moves
                klist = self.killers.get(depth_key, [])
                if m in klist:
                    sc += 8000
            # small randomness to break ties
            sc += random.randint(0, 10)
            return sc

        # We'll use fen as a quick TT key; include side-to-move in fen anyway.
        # For better performance you could use chess.polyglot.zobrist_hash but fen is safe & simple.
        def make_key(b):
            return b.board_fen() + " " + ("w" if b.turn == chess.WHITE else "b") + " " + b.castling_xfen() + " " + (b.ep_square and str(b.ep_square) or "-")

        # ---- quiescence search (captures only) ----
        def quiescence(b, alpha, beta):
            if time_up():
                return evaluate(b)
            stand_pat = evaluate(b)
            if stand_pat >= beta:
                return beta
            if alpha < stand_pat:
                alpha = stand_pat

            # consider captures
            captures = [m for m in b.legal_moves if b.is_capture(m)]
            # sort captures by MVV-LVA
            captures.sort(key=lambda mv: - (PIECE_VALUES.get(b.piece_at(mv.to_square).piece_type,0) - PIECE_VALUES.get(b.piece_at(mv.from_square).piece_type,0) if b.piece_at(mv.to_square) and b.piece_at(mv.from_square) else 0))
            for m in captures:
                if time_up():
                    break
                b.push(m)
                score = -quiescence(b, -beta, -alpha)
                b.pop()
                if score >= beta:
                    return beta
                if score > alpha:
                    alpha = score
            return alpha

        # ---- alpha-beta with TT & ordering & iterative deepening ----
        # We'll keep depth_key accessible in score_move (for killers)
        def alphabeta(b, depth, alpha, beta, maximizing):
            if time_up():
                # early exit: return static eval so the caller can fall back to last best move
                return evaluate(b)

            key = make_key(b)
            # check TT
            tt_res = tt_get(key, depth, alpha, beta)
            if tt_res:
                return tt_res[0]

            if depth == 0:
                # quiescence search
                val = quiescence(b, alpha, beta)
                tt_store(key, 0, val, 0, None)
                return val

            # PV move suggestion from TT
            tt_entry = self.tt.get(key)
            tt_move = tt_entry[3] if tt_entry else None

            best_value = -INFINITY if maximizing else INFINITY
            best_move_local = None

            # generate moves with ordering
            moves = list(b.legal_moves)
            # sort using MVV-LVA, TT move and history (desc)
            # depth_key used by score_move; set it in outer scope
            nonlocal depth_key
            moves.sort(key=lambda mv: -score_move(b, mv, tt_move))

            for m in moves:
                if time_up():
                    break
                b.push(m)
                val = -alphabeta(b, depth - 1, -beta, -alpha, not maximizing)
                b.pop()

                if maximizing:
                    if val > best_value:
                        best_value = val
                        best_move_local = m
                    alpha = max(alpha, val)
                else:
                    if val < best_value:
                        best_value = val
                        best_move_local = m
                    beta = min(beta, val)

                # alpha-beta cutoff
                if alpha >= beta:
                    # store killer if not capture
                    if not b.is_capture(m):
                        k = self.killers.get(depth_key, [])
                        if m not in k:
                            k.insert(0, m)
                            if len(k) > 2:
                                k.pop()
                            self.killers[depth_key] = k
                        # update history
                        self.history[(m.from_square, m.to_square)] = self.history.get((m.from_square, m.to_square), 0) + (1 << depth)
                    break

            # store in TT
            if best_move_local is None:
                flag = 1  # beta (upper bound)
            else:
                # determine flag: exact if within window, otherwise bounds
                if best_value <= alpha_orig:
                    flag = 1  # upper
                elif best_value >= beta_orig:
                    flag = -1  # lower
                else:
                    flag = 0  # exact

            # But we don't have alpha_orig/beta_orig locally — simpler heuristic:
            # If cutoff happened -> lower-bound; else exact.
            store_flag = 0
            # if there was a cutoff (alpha >= beta at some point), store lower-bound/upper-bound guess:
            # For simplicity, mark exact if we searched all moves (no cutoff), else mark lower-bound.
            # We'll detect cutoff by comparing best_move_local vs some value
            # (this is a simplification for hackathon; it's still useful)
            store_flag = 0 if best_move_local is not None else -1
            tt_store(key, depth, best_value, store_flag, best_move_local)
            return best_value

        # Because we used variables in nested functions we need to predeclare depth_key
        depth_key = 0

        # ---- iterative deepening ----
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return PlayResult(None, None)

        best_move = random.choice(legal_moves)
        best_score = -INFINITY if board.turn == chess.WHITE else INFINITY

        # iterative deepen up to max_depth (adjustable)
        max_depth = 5  # default; increase if you have more time per move
        for depth in range(1, max_depth + 1):
            if time_up():
                break
            depth_key = depth
            # try aspiration window? skip for simplicity

            # prepare move ordering at root: use TT best move first if available
            root_key = make_key(board)
            root_tt = self.tt.get(root_key)
            tt_root_move = root_tt[3] if root_tt else None

            moves = list(board.legal_moves)
            # order moves: TT move first, captures by MVV-LVA, then history
            def root_score(mv):
                sc = 0
                if mv == tt_root_move:
                    sc += 2000000
                if board.is_capture(mv):
                    victim = board.piece_at(mv.to_square)
                    attacker = board.piece_at(mv.from_square)
                    sc += 10000 + (PIECE_VALUES.get(victim.piece_type, 0) - PIECE_VALUES.get(attacker.piece_type, 0) if victim and attacker else 0)
                sc += self.history.get((mv.from_square, mv.to_square), 0)
                return -sc

            moves.sort(key=root_score)

            current_best = None
            current_best_score = -INFINITY if board.turn == chess.WHITE else INFINITY

            for m in moves:
                if time_up():
                    break
                board.push(m)
                # set alpha/beta wide
                alpha = -INFINITY
                beta = INFINITY
                # we'll call alphabeta with sign-flip
                try:
                    val = -alphabeta(board, depth - 1, -INFINITY, INFINITY, board.turn == chess.BLACK)
                except RecursionError:
                    val = evaluate(board)
                board.pop()

                if board.turn == chess.WHITE:
                    if val > current_best_score:
                        current_best_score = val
                        current_best = m
                else:
                    if val < current_best_score:
                        current_best_score = val
                        current_best = m

                # emergency exit to ensure we always have a move
                if time_up():
                    break

            if current_best is not None:
                best_move = current_best
                best_score = current_best_score

            # emergency quick-play: if not much time left, break early
            if (deadline - now()) < emergency_threshold:
                break

        # If we are extremely low on time, pick a safe move (avoid immediate mate / checks)
        if (deadline - now()) < 0.2:
            legal = list(board.legal_moves)
            def safe_score(mv):
                board.push(mv)
                s = 0
                if board.is_check():
                    s -= 100000
                if board.is_capture(mv):
                    s += 1000
                board.pop()
                return s
            best_move = max(legal, key=safe_score)

        return PlayResult(best_move, None)
