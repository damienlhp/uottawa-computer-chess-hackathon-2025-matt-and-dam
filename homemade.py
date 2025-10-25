import time
import random
import chess
from chess.engine import PlayResult, Limit
from lib.engine_wrapper import MinimalEngine

print('chess bot version 6.0')

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

INFINITY = 10**9


class StrongBot(MinimalEngine):
    """Fast version: consistent <2s per move, shallow but smart.

    Accept the same constructor signature as `MinimalEngine` so the
    engine factory can instantiate this class with kwargs like `cwd`.
    """

    def __init__(self, commands=None, options=None, stderr=None, draw_or_resign=None,
                 game=None, name=None, **popen_args):
        # Forward parameters to MinimalEngine which will create a FillerEngine
        # for homemade engines. Keep local engine data structures after init.
        super().__init__(commands, options, stderr, draw_or_resign, game, name=name, **popen_args)
        self.tt = {}
        self.history = {}
        self.killers = {}

    def evaluate(self, b: chess.Board) -> int:
        """Simple but quick evaluation (material + basic king safety)."""
        if b.is_game_over():
            outcome = b.outcome()
            if outcome is None or outcome.winner is None:
                return 0
            return INFINITY if outcome.winner == chess.WHITE else -INFINITY

        score = 0
        for piece_type, val in PIECE_VALUES.items():
            score += val * (len(b.pieces(piece_type, chess.WHITE)) -
                            len(b.pieces(piece_type, chess.BLACK)))

        # Small king safety term (avoid checks)
        if b.is_check():
            score += -50 if b.turn == chess.WHITE else 50

        return score

    def search(self, board: chess.Board, *args) -> PlayResult:
        """Main search — tuned for fast, consistent response."""
        # --- time control: never exceed 1.5–2s per move ---
        time_limit = args[0] if (args and isinstance(args[0], Limit)) else None
        max_time = 1.2
        if isinstance(time_limit, Limit) and isinstance(time_limit.time, (int, float)):
            # adapt a little, but stay fast
            max_time = max(0.8, min(1.8, time_limit.time / 60.0))

        start = time.time()
        deadline = start + max_time

        def time_up():
            return time.time() >= deadline

        def quiescence(b, alpha, beta):
            """Capture-only extension."""
            stand_pat = self.evaluate(b)
            if stand_pat >= beta:
                return beta
            if alpha < stand_pat:
                alpha = stand_pat

            for m in b.legal_moves:
                if not b.is_capture(m):
                    continue
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

        def alphabeta(b, depth, alpha, beta):
            if depth == 0 or time_up():
                return quiescence(b, alpha, beta)

            best_val = -INFINITY
            for m in b.legal_moves:
                b.push(m)
                val = -alphabeta(b, depth - 1, -beta, -alpha)
                b.pop()

                if val > best_val:
                    best_val = val
                if best_val > alpha:
                    alpha = best_val
                if alpha >= beta or time_up():
                    break
            return best_val

        # --- fast iterative deepening but capped at depth 3 ---
        best_move = random.choice(list(board.legal_moves))
        best_score = -INFINITY
        for depth in range(1, 4):  # fixed shallow depth
            if time_up():
                break
            current_best = None
            current_best_score = -INFINITY
            for m in board.legal_moves:
                if time_up():
                    break
                board.push(m)
                val = -alphabeta(board, depth - 1, -INFINITY, INFINITY)
                board.pop()
                if val > current_best_score:
                    current_best_score = val
                    current_best = m
            if current_best:
                best_move, best_score = current_best, current_best_score

        # --- emergency fallback if time is up ---
        if time_up():
            legal = list(board.legal_moves)
            safe = [m for m in legal if not board.gives_check(m)]
            if safe:
                best_move = random.choice(safe)
            else:
                best_move = random.choice(legal)

        print(f"🕐 Fast move made in {time.time() - start:.2f}s")
        return PlayResult(best_move, None)


class ExampleEngine(MinimalEngine):
    """Backward-compatible example engine used by tests.

    Provides a simple constructor matching the engine factory expectations
    and a trivial `search` implementation that returns a legal move.
    """

    def __init__(self, commands, options, stderr, draw_or_resign, game=None, **popen_args):
        # MinimalEngine expects (commands, options, stderr, draw_or_resign, game=None, name=None, **popen_args)
        # We forward parameters and let the wrapper provide a FillerEngine.
        super().__init__(commands, options, stderr, draw_or_resign, game, **popen_args)

    def search(self, board: chess.Board, time_limit: Limit, ponder: bool, draw_offered: bool, root_moves):
        # Return the first legal move if available, else None.
        mv = next(board.legal_moves, None)
        return self.offer_draw_or_resign(PlayResult(mv, None), board)


# Backwards compatibility: some configurations and tests expect a class named
# `MyBot` in this module. Provide a simple alias so `get_homemade_engine("MyBot")`
# and imports like `from homemade import MyBot` continue to work.
MyBot = StrongBot
