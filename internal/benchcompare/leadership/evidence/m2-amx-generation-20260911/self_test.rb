#!/usr/bin/env ruby
# frozen_string_literal: true

require "fileutils"
require "tmpdir"
require_relative "fixture"

module AMXGenerationEvidenceSelfTest
  module_function

  def check(condition, message)
    raise "self-test failed: #{message}" unless condition
  end

  def invalid(name)
    begin
      yield
    rescue AMXGenerationEvidence::Invalid
      return
    end
    raise "self-test failed: #{name} was accepted"
  end

  def write_json(path, object)
    File.binwrite(path, JSON.generate(object))
  end

  def mutate_json(path)
    object = JSON.parse(File.binread(path))
    yield object
    write_json(path, object)
  end

  def rewrite_stream(dir, name, text)
    File.binwrite(File.join(dir, name), text)
    result_path = File.join(dir, "result.json")
    mutate_json(result_path) do |result|
      result["streams"][name] = {
        "bytes" => text.bytesize,
        "sha256" => Digest::SHA256.hexdigest(text)
      }
    end
  end

  def stdout_text(arm, campaign, pair)
    lines = AMXGenerationEvidence::STDOUT_PREFIX.dup
    AMXGenerationEvidence::CASES.each_with_index do |name, index|
      iterations = 1000 + campaign * 100 + pair * 10 + index
      latency = 2000 + campaign * 100 + pair * 10 + index + (arm == "candidate" ? 1 : 0)
      bytes = arm == "candidate" ? 288 : 272
      lines << "#{name} #{iterations} #{latency} ns/op #{bytes} B/op 4 allocs/op"
    end
    lines << "PASS"
    lines.join("\n") + "\n"
  end

  def create_root(parent, revision: "candidate1", campaigns: AMXGenerationEvidence::CAMPAIGNS)
    root = File.join(parent, "root-#{Dir.children(parent).length}")
    FileUtils.mkdir_p(root)
    runner = File.join(root, "runner.rb")
    binaries = { "baseline" => File.join(root, "baseline.test"), "candidate" => File.join(root, "candidate.test") }
    File.binwrite(runner, "synthetic runner\n")
    File.binwrite(binaries["baseline"], "synthetic baseline\n")
    File.binwrite(binaries["candidate"], "synthetic candidate\n")
    plan = AMXGenerationEvidence::Plan.new(
      revision: revision,
      runner: File.basename(runner),
      runner_sha256: Digest::SHA256.file(runner).hexdigest,
      binaries: binaries.transform_values { |path| File.basename(path) },
      binary_sha256: binaries.transform_values { |path| Digest::SHA256.file(path).hexdigest }
    )
    workdir = File.join(root, "workdir")
    FileUtils.mkdir_p(workdir)
    base = Time.utc(2026, 9, 11, 1, 0, 0)
    sequence = 0
    campaigns.each do |campaign|
      campaign_dir = File.join(root, format("campaign-%02d", campaign))
      FileUtils.mkdir_p(campaign_dir)
      first_start = base + sequence * 10 + 2
      manifest = {
        "protocol" => AMXGenerationEvidence::PROTOCOL,
        "campaign" => campaign,
        "arms" => AMXGenerationEvidence::ARMS.to_h { |arm| [arm, [binaries.fetch(arm), plan.binary_sha256.fetch(arm)]] },
        "cases" => AMXGenerationEvidence::CASES,
        "pairs" => AMXGenerationEvidence::PAIRS,
        "excluded_pairs" => [0],
        "benchtime" => "500ms",
        "gomaxprocs" => 12,
        "runner_sha256" => plan.runner_sha256,
        "reviewed_preflight_sha256" => "a" * 64,
        "reviewed_preflight_ended_at" => (first_start - 1).iso8601(6)
      }
      write_json(File.join(campaign_dir, "manifest.json"), manifest)
      AMXGenerationEvidence::PAIRS.each do |pair|
        AMXGenerationEvidence.expected_order(campaign, pair).each do |arm|
          sequence += 1
          started = base + sequence * 10
          ended = started + 1
          dir = File.join(campaign_dir, format("%02d-%s", pair, arm))
          FileUtils.mkdir_p(dir)
          argv = AMXGenerationEvidence::EXPECTED_ENV + [binaries.fetch(arm)] + AMXGenerationEvidence::EXPECTED_ARGV_TAIL
          command = { "argv" => argv, "cwd" => workdir, "started_at" => started.iso8601(6) }
          stdout = stdout_text(arm, campaign, pair)
          stderr = ""
          result = command.merge(
            "ended_at" => ended.iso8601(6),
            "exit" => 0,
            "signal" => nil,
            "streams" => {
              "stdout" => { "bytes" => stdout.bytesize, "sha256" => Digest::SHA256.hexdigest(stdout) },
              "stderr" => { "bytes" => 0, "sha256" => AMXGenerationEvidence::EMPTY_SHA256 }
            },
            "capture_helper_sha256" => AMXGenerationEvidence::CAPTURE_HELPER_SHA256
          )
          write_json(File.join(dir, "command.json"), command)
          write_json(File.join(dir, "result.json"), result)
          File.binwrite(File.join(dir, "stdout"), stdout)
          File.binwrite(File.join(dir, "stderr"), stderr)
        end
      end
    end
    [plan, root]
  end

  def fresh(parent, revision: "candidate1", campaigns: AMXGenerationEvidence::CAMPAIGNS)
    create_root(parent, revision: revision, campaigns: campaigns)
  end

  def capture(root, campaign: 1, pair: 0, arm: "baseline")
    File.join(root, format("campaign-%02d", campaign), format("%02d-%s", pair, arm))
  end

  def csv_mutation(base)
    table = CSV.parse(base, headers: true)
    yield table
    CSV.generate(row_sep: "\n") do |csv|
      csv << table.headers
      table.each { |row| csv << row.fields }
    end
  end

  def run
    Dir.mktmpdir("amx-generation-evidence-self-test") do |tmp|
      plan, root = fresh(tmp)
      rows = AMXGenerationEvidence.build_rows(plan, root)
      check(rows.length == 240, "positive synthetic plan row count")
      base = AMXGenerationEvidence.csv_text(rows)
      check(AMXGenerationEvidence.parse_csv(base).length == 240, "positive CSV validation")
      check(AMXGenerationEvidence.summary_csv(AMXGenerationEvidence.parse_csv(base)).lines.length == 31, "summary shape")
      optional_plan, optional_root = fresh(tmp)
      %w[samples.csv baseline.bench candidate.bench].each { |name| File.binwrite(File.join(optional_root, "campaign-01", name), "derived\n") }
      check(AMXGenerationEvidence.build_rows(optional_plan, optional_root).length == 240, "known derived campaign artifacts")

      first = File.join(tmp, "first.csv")
      second = File.join(tmp, "second.csv")
      AMXGenerationEvidence.build_csv(plan, root, first)
      AMXGenerationEvidence.build_csv(plan, root, second)
      check(File.binread(first) == File.binread(second), "two builds are not byte-identical")

      candidate2 = base.sub(/^candidate1,/, "candidate2,").gsub(/\ncandidate1,/, "\ncandidate2,")
      check(AMXGenerationEvidence.parse_csv(candidate2, expected_revision: "candidate2").length == 240, "candidate2 CSV validation")

      csv_cases = {
        "missing row" => ->(table) { table.delete(table.length - 1) },
        "duplicate replacing a cell" => ->(table) { table[1] = table[0] },
        "unknown revision" => ->(table) { table[0]["revision"] = "candidate3" },
        "mixed revision" => ->(table) { table[0]["revision"] = "candidate2" },
        "unknown campaign" => ->(table) { table[0]["campaign"] = "4" },
        "unknown pair" => ->(table) { table[0]["pair"] = "8" },
        "wrong phase" => ->(table) { table[0]["phase"] = "measured" },
        "unknown arm" => ->(table) { table[0]["arm"] = "control" },
        "unknown case" => ->(table) { table[0]["case"] = "BenchmarkOther/x-12" },
        "zero iterations" => ->(table) { table[0]["iterations"] = "0" },
        "negative iterations" => ->(table) { table[0]["iterations"] = "-1" },
        "fractional iterations" => ->(table) { table[0]["iterations"] = "1.5" },
        "huge iterations" => ->(table) { table[0]["iterations"] = "9007199254740992" },
        "zero latency" => ->(table) { table[0]["ns_per_op"] = "0" },
        "negative latency" => ->(table) { table[0]["ns_per_op"] = "-1" },
        "NaN latency" => ->(table) { table[0]["ns_per_op"] = "NaN" },
        "infinite latency" => ->(table) { table[0]["ns_per_op"] = "Infinity" },
        "overflow latency" => ->(table) { table[0]["ns_per_op"] = "1e999" },
        "fractional bytes" => ->(table) { table[0]["bytes_per_op"] = "1.5" },
        "negative bytes" => ->(table) { table[0]["bytes_per_op"] = "-1" },
        "huge bytes" => ->(table) { table[0]["bytes_per_op"] = "9007199254740992" },
        "fractional allocations" => ->(table) { table[0]["allocs_per_op"] = "1.5" },
        "negative allocations" => ->(table) { table[0]["allocs_per_op"] = "-1" },
        "huge allocations" => ->(table) { table[0]["allocs_per_op"] = "9007199254740992" }
      }
      csv_cases.each do |name, mutation|
        invalid(name) { AMXGenerationEvidence.parse_csv(csv_mutation(base, &mutation)) }
      end
      invalid("missing CSV column") { AMXGenerationEvidence.parse_csv(base.sub("allocs_per_op", "")) }
      invalid("extra CSV column") { AMXGenerationEvidence.parse_csv(base.sub("allocs_per_op", "allocs_per_op,extra")) }
      invalid("extra row field") { AMXGenerationEvidence.parse_csv(base.sub(/\n([^\n]+)/, "\n\\1,extra")) }
      invalid("missing row field") { AMXGenerationEvidence.parse_csv(base.sub(/,4\n/, "\n")) }
      huge_summary = csv_mutation(base) do |table|
        group = table.select do |row|
          row["campaign"] == "1" && row["phase"] == "measured" && row["arm"] == "baseline" && row["case"] == AMXGenerationEvidence::CASES.first
        end
        group.each { |row| row["ns_per_op"] = "1" }
        group.first["ns_per_op"] = "17" + "0" * 307
      end
      invalid("nonfinite derived summary") { AMXGenerationEvidence.summary_csv(AMXGenerationEvidence.parse_csv(huge_summary)) }

      builder_cases = {
        "runner content" => lambda do |p, r|
          File.binwrite(File.join(r, p.runner), "tampered\n")
        end,
        "binary content" => lambda do |p, r|
          File.binwrite(File.join(r, p.binaries.fetch("candidate")), "tampered\n")
        end,
        "manifest extra key" => lambda do |_p, r|
          mutate_json(File.join(r, "campaign-01", "manifest.json")) { |m| m["extra"] = true }
        end,
        "manifest protocol" => lambda do |_p, r|
          mutate_json(File.join(r, "campaign-01", "manifest.json")) { |m| m["protocol"] = "other" }
        end,
        "manifest benchtime" => lambda do |_p, r|
          mutate_json(File.join(r, "campaign-01", "manifest.json")) { |m| m["benchtime"] = "1s" }
        end,
        "manifest numeric types" => lambda do |_p, r|
          mutate_json(File.join(r, "campaign-01", "manifest.json")) { |m| m["gomaxprocs"] = 12.0 }
        end,
        "manifest pair numeric types" => lambda do |_p, r|
          mutate_json(File.join(r, "campaign-01", "manifest.json")) { |m| m["pairs"][0] = 0.0 }
        end,
        "manifest cases" => lambda do |_p, r|
          mutate_json(File.join(r, "campaign-01", "manifest.json")) { |m| m["cases"] = m["cases"][0, 4] }
        end,
        "manifest binary path" => lambda do |_p, r|
          mutate_json(File.join(r, "campaign-01", "manifest.json")) { |m| m["arms"]["candidate"][0] = m["arms"]["baseline"][0] }
        end,
        "capture extra file" => lambda do |_p, r|
          File.binwrite(File.join(capture(r), "extra"), "x")
        end,
        "command/result disagreement" => lambda do |_p, r|
          mutate_json(File.join(capture(r), "command.json")) { |m| m["started_at"] = "2026-09-11T01:00:00.000000Z" }
        end,
        "malformed result JSON" => lambda do |_p, r|
          File.binwrite(File.join(capture(r), "result.json"), "{")
        end,
        "argv environment" => lambda do |_p, r|
          ["command.json", "result.json"].each { |name| mutate_json(File.join(capture(r), name)) { |m| m["argv"][1] = "GOMAXPROCS=11" } }
        end,
        "argv binary" => lambda do |_p, r|
          ["command.json", "result.json"].each { |name| mutate_json(File.join(capture(r), name)) { |m| m["argv"][5] = m["argv"][5] + ".other" } }
        end,
        "argv test flag" => lambda do |_p, r|
          ["command.json", "result.json"].each { |name| mutate_json(File.join(capture(r), name)) { |m| m["argv"][-3] = "-test.benchtime=1s" } }
        end,
        "relative cwd" => lambda do |_p, r|
          ["command.json", "result.json"].each { |name| mutate_json(File.join(capture(r), name)) { |m| m["cwd"] = "." } }
        end,
        "invalid calendar timestamp" => lambda do |_p, r|
          ["command.json", "result.json"].each { |name| mutate_json(File.join(capture(r), name)) { |m| m["started_at"] = "2026-02-30T01:00:00Z" } }
        end,
        "end before start" => lambda do |_p, r|
          mutate_json(File.join(capture(r), "result.json")) { |m| m["ended_at"] = "2026-09-11T00:00:00.000000Z" }
        end,
        "overlapping captures" => lambda do |_p, r|
          dir = capture(r, arm: "candidate")
          previous = JSON.parse(File.binread(File.join(capture(r), "result.json")))["started_at"]
          ["command.json", "result.json"].each { |name| mutate_json(File.join(dir, name)) { |m| m["started_at"] = previous } }
        end,
        "out-of-order capture" => lambda do |_p, r|
          dir = capture(r, arm: "candidate")
          ["command.json", "result.json"].each { |name| mutate_json(File.join(dir, name)) { |m| m["started_at"] = "2026-09-11T00:00:01.000000Z" } }
          mutate_json(File.join(dir, "result.json")) { |m| m["ended_at"] = "2026-09-11T00:00:02.000000Z" }
        end,
        "stale preflight" => lambda do |_p, r|
          mutate_json(File.join(r, "campaign-01", "manifest.json")) { |m| m["reviewed_preflight_ended_at"] = "2026-09-10T00:00:00.000000Z" }
        end,
        "preflight after launch" => lambda do |_p, r|
          mutate_json(File.join(r, "campaign-01", "manifest.json")) { |m| m["reviewed_preflight_ended_at"] = "2027-09-11T00:00:00.000000Z" }
        end,
        "preflight hash" => lambda do |_p, r|
          mutate_json(File.join(r, "campaign-01", "manifest.json")) { |m| m["reviewed_preflight_sha256"] = "A" * 64 }
        end,
        "nonzero exit" => lambda do |_p, r|
          mutate_json(File.join(capture(r), "result.json")) { |m| m["exit"] = 1 }
        end,
        "noninteger exit" => lambda do |_p, r|
          mutate_json(File.join(capture(r), "result.json")) { |m| m["exit"] = 0.0 }
        end,
        "signal" => lambda do |_p, r|
          mutate_json(File.join(capture(r), "result.json")) { |m| m["signal"] = "TERM" }
        end,
        "capture helper" => lambda do |_p, r|
          mutate_json(File.join(capture(r), "result.json")) { |m| m["capture_helper_sha256"] = "b" * 64 }
        end,
        "stdout content hash mismatch" => lambda do |_p, r|
          File.open(File.join(capture(r), "stdout"), "ab") { |file| file.write("x") }
        end,
        "stdout declared size" => lambda do |_p, r|
          mutate_json(File.join(capture(r), "result.json")) { |m| m["streams"]["stdout"]["bytes"] += 1 }
        end,
        "stdout declared hash" => lambda do |_p, r|
          mutate_json(File.join(capture(r), "result.json")) { |m| m["streams"]["stdout"]["sha256"] = "c" * 64 }
        end,
        "nonempty stderr" => lambda do |_p, r|
          rewrite_stream(capture(r), "stderr", "warning\n")
        end,
        "missing PASS" => lambda do |_p, r|
          rewrite_stream(capture(r), "stdout", stdout_text("baseline", 1, 0).sub("PASS\n", ""))
        end,
        "duplicate PASS" => lambda do |_p, r|
          rewrite_stream(capture(r), "stdout", stdout_text("baseline", 1, 0) + "PASS\n")
        end,
        "FAIL marker" => lambda do |_p, r|
          rewrite_stream(capture(r), "stdout", stdout_text("baseline", 1, 0).sub("PASS", "FAIL"))
        end,
        "SKIP marker" => lambda do |_p, r|
          rewrite_stream(capture(r), "stdout", stdout_text("baseline", 1, 0).sub("PASS", "SKIP\nPASS"))
        end,
        "missing benchmark" => lambda do |_p, r|
          text = stdout_text("baseline", 1, 0)
          rewrite_stream(capture(r), "stdout", text.lines.reject { |line| line.start_with?(AMXGenerationEvidence::CASES.last) }.join)
        end,
        "duplicate benchmark" => lambda do |_p, r|
          text = stdout_text("baseline", 1, 0)
          line = text.lines.find { |item| item.start_with?(AMXGenerationEvidence::CASES.first) }
          rewrite_stream(capture(r), "stdout", text.sub("PASS\n", line + "PASS\n"))
        end,
        "same-count duplicate case and omission" => lambda do |_p, r|
          text = stdout_text("baseline", 1, 0).sub(AMXGenerationEvidence::CASES[1], AMXGenerationEvidence::CASES[0])
          rewrite_stream(capture(r), "stdout", text)
        end,
        "extra benchmark metric" => lambda do |_p, r|
          text = stdout_text("baseline", 1, 0).sub(" ns/op", " ns/op 99 MB/s")
          rewrite_stream(capture(r), "stdout", text)
        end,
        "duplicate metric with unchanged field count" => lambda do |_p, r|
          text = stdout_text("baseline", 1, 0).sub(" B/op", " ns/op")
          rewrite_stream(capture(r), "stdout", text)
        end,
        "invalid benchmark latency" => lambda do |_p, r|
          text = stdout_text("baseline", 1, 0).sub(/ [0-9]+ ns\/op/, " NaN ns/op")
          rewrite_stream(capture(r), "stdout", text)
        end,
        "unexpected platform preamble" => lambda do |_p, r|
          text = stdout_text("baseline", 1, 0).sub("cpu: Apple M2 Pro", "cpu: another host")
          rewrite_stream(capture(r), "stdout", text)
        end
      }
      builder_cases.each do |name, mutation|
        test_plan, test_root = fresh(tmp)
        mutation.call(test_plan, test_root)
        invalid(name) { AMXGenerationEvidence.build_rows(test_plan, test_root) }
      end

      test_plan, test_root = fresh(tmp, campaigns: [1, 2])
      invalid("incomplete campaign set") { AMXGenerationEvidence.build_rows(test_plan, test_root) }
      test_plan, test_root = fresh(tmp)
      FileUtils.mkdir_p(File.join(test_root, "campaign-04"))
      invalid("extra campaign") { AMXGenerationEvidence.build_rows(test_plan, test_root) }
      test_plan, test_root = fresh(tmp)
      FileUtils.rm_r(capture(test_root))
      invalid("missing invocation") { AMXGenerationEvidence.build_rows(test_plan, test_root) }
      test_plan, test_root = fresh(tmp)
      FileUtils.rm_f(File.join(capture(test_root), "command.json"))
      invalid("missing command metadata") { AMXGenerationEvidence.build_rows(test_plan, test_root) }
      test_plan, test_root = fresh(tmp)
      FileUtils.mkdir_p(File.join(test_root, "campaign-01", "08-baseline"))
      invalid("extra invocation") { AMXGenerationEvidence.build_rows(test_plan, test_root) }
      test_plan, test_root = fresh(tmp)
      File.binwrite(File.join(test_root, "campaign-01", "unknown.txt"), "x")
      invalid("unknown campaign artifact") { AMXGenerationEvidence.build_rows(test_plan, test_root) }

      empty_plan, empty_root = fresh(tmp, revision: "candidate2", campaigns: [])
      invalid("incomplete candidate2") { AMXGenerationEvidence.build_rows(empty_plan, empty_root) }

      output_plan, output_root = fresh(tmp)
      existing = File.join(tmp, "existing.csv")
      File.binwrite(existing, "preserve me")
      invalid("existing output") { AMXGenerationEvidence.build_csv(output_plan, output_root, existing) }
      check(File.binread(existing) == "preserve me", "existing output changed")
      dangling = File.join(tmp, "dangling.csv")
      File.symlink(File.join(tmp, "missing-target"), dangling)
      invalid("dangling output symlink") { AMXGenerationEvidence.build_csv(output_plan, output_root, dangling) }
      check(File.symlink?(dangling), "dangling output symlink changed")
      partial = File.join(tmp, "partial.csv")
      FileUtils.rm_r(capture(output_root))
      invalid("partial output") { AMXGenerationEvidence.build_csv(output_plan, output_root, partial) }
      check(!File.exist?(partial), "failed build left partial output")
    end
    puts "self-test: all adversarial and positive cases passed"
  end
end

AMXGenerationEvidenceSelfTest.run
