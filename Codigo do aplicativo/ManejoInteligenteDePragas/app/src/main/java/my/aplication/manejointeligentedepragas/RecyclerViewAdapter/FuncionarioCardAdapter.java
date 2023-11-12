package my.aplication.manejointeligentedepragas.RecyclerViewAdapter;

import android.app.AlertDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
//import android.support.annotation.NonNull;
import androidx.annotation.NonNull;
//import android.support.v7.widget.RecyclerView;
import androidx.recyclerview.widget.RecyclerView;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.RelativeLayout;
import android.widget.TextView;
import android.widget.Toast;

import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;
import com.example.manejointeligentedepragas.R;
import my.aplication.manejointeligentedepragas.VisualizaFuncionario;
import my.aplication.manejointeligentedepragas.model.UsuarioModel;

import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;

public class FuncionarioCardAdapter extends RecyclerView.Adapter<FuncionarioCardAdapter.ViewHolder>{

    private static final String TAG = "PropriedadeCardAdapter";

    private ArrayList<UsuarioModel> cards = new ArrayList<>();
    private Context funcionarioContext;
    private int Cod_Propriedade;
    private String nomePropriedade;

    public FuncionarioCardAdapter(Context funcionarioContext, ArrayList<UsuarioModel> cards, int Cod_Propriedade, String nomePropriedade) {
        this.cards = cards;
        this.funcionarioContext = funcionarioContext;
        this.Cod_Propriedade = Cod_Propriedade;
        this.nomePropriedade = nomePropriedade;
    }

    @NonNull
    @Override
    // resposável por inflar a view

    public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {

        View view = LayoutInflater.from(parent.getContext()).inflate(R.layout.card_funcionario, parent, false);
        ViewHolder holder = new ViewHolder(view);
        return holder;
    }

    @Override
    public void onBindViewHolder(@NonNull ViewHolder holder, final int position) {
        Log.d(TAG, "onBindViewHolder: called.");

        holder.nome.setText(cards.get(position).getNome());
        holder.email.setText("E-mail: " + cards.get(position).getEmail());
        holder.telefone.setText("Telefone: " + cards.get(position).getTelefone());


        holder.parent_layout_funcionario.setOnLongClickListener(new View.OnLongClickListener() {
            @Override
            public boolean onLongClick(View v) {
                exibirCaixaConfirmacao(cards.get(position));
                return true;
            }
        });

    }

    @Override
    public int getItemCount() {
        return cards.size();
    }

    public class ViewHolder extends RecyclerView.ViewHolder {

        TextView nome;
        TextView email;
        TextView telefone;

        RelativeLayout parent_layout_funcionario;


        public ViewHolder(@NonNull View itemView) {
            super(itemView);

            nome = itemView.findViewById(R.id.tvNomeFuncionario);
            email = itemView.findViewById(R.id.tvEmailFunc);
            telefone = itemView.findViewById(R.id.tvTelefoneFunc);

            parent_layout_funcionario = itemView.findViewById(R.id.parent_layout_funcionario);
        }


    }

    public void exibirCaixaConfirmacao(final UsuarioModel F)
    {
        AlertDialog.Builder dlgBox = new AlertDialog.Builder(funcionarioContext);
        dlgBox.setTitle("Removendo...");
        dlgBox.setMessage("Tem certeza que deseja remover o funcionário dessa propriedade?");
        dlgBox.setPositiveButton("Sim", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                ExcluirFuncionario(F.getCod_Usuario());
            }
        });

        dlgBox.setNegativeButton("Não", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                // não faz nenhuma ação
            }
        });

        dlgBox.show();
    }


    public void ExcluirFuncionario(Integer CodU){

        String url = "https://mip.software/phpapp/excluirFuncionario.php?Cod_Usuario="+CodU+"&&Cod_Propriedade="+Cod_Propriedade ;

        RequestQueue queue = Volley.newRequestQueue(funcionarioContext);
        queue.add(new StringRequest(Request.Method.POST, url, new Response.Listener<String>() {

            @Override
            public void onResponse(String response) {
                //Parsing json
                //Toast.makeText(Entrar.this,"AQUI", Toast.LENGTH_LONG).show();
                try {
                    JSONObject obj = new JSONObject(response);
                    boolean confirmacao = obj.getBoolean("confirmacao");
                    if(confirmacao){
                        Intent k = new Intent(funcionarioContext, VisualizaFuncionario.class);
                        Toast.makeText(funcionarioContext, "Funcionario removido da propriedade!",Toast.LENGTH_LONG).show();
                        k.putExtra("Cod_Propriedade", Cod_Propriedade);
                        k.putExtra("nomePropriedade", nomePropriedade);
                        funcionarioContext.startActivity(k);
                    }else{
                        Toast.makeText(funcionarioContext, "Funcionário não removido! Tente novamente",Toast.LENGTH_LONG).show();
                    }

                } catch (JSONException e) {
                    Toast.makeText(funcionarioContext, e.toString(), Toast.LENGTH_LONG).show();
                }
            }
        }, new Response.ErrorListener() {
            @Override
            public void onErrorResponse(VolleyError error) {
                Toast.makeText(funcionarioContext,error.toString(), Toast.LENGTH_LONG).show();
            }
        }));
    }

}
